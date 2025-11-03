import torch
import asyncio
import logging
from enum import Enum
from typing import Dict, Any
import gc

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enum for different model types"""
    BINARY_CLASSIFIER = "binary_classifier"
    MULTICLASS_CLASSIFIER = "multiclass_classifier"
    SEGMENTATION = "segmentation"
    RAG = "rag"
    REPORT_GENERATOR = "report_generator"
    FEATURE_EXTRACTOR = "feature_extractor"


class ModelRegistry:
    """
    Central registry for managing multiple models
    
    Features:
    - Lazy loading: Models are loaded only when needed
    - Memory management: Can unload models to free memory
    - Caching: Keep frequently used models in memory
    - Thread-safe: Async-compatible for concurrent requests
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model registry
        
        Args:
            config: Configuration dictionary with model settings
        """
        self.config = config
        self.models: Dict[ModelType, Any] = {}
        self.model_configs = config.get("models", {})
        self.cache_models = config.get("cache_models", True)
        self.device = torch.device(config.get("device", "cuda") 
                                   if torch.cuda.is_available() else "cpu")
        self._locks: Dict[ModelType, asyncio.Lock] = {
            model_type: asyncio.Lock() for model_type in ModelType
        }
        
        logger.info(f"Model registry initialized with device: {self.device}")
    
    async def get_model(self, model_type: ModelType) -> Any:
        """
        Get a model instance, loading it if necessary
        
        Args:
            model_type: Type of model to retrieve
            
        Returns:
            Model adapter instance
            
        Raises:
            ValueError: If model type is not enabled in config
            RuntimeError: If model fails to load
        """
        async with self._locks[model_type]:
            # Check if model is already loaded
            if model_type in self.models:
                logger.debug(f"Using cached model: {model_type.value}")
                return self.models[model_type]
            
            # Check if model is enabled
            model_config_key = model_type.value
            if model_config_key not in self.model_configs:
                raise ValueError(f"Model {model_type.value} not found in configuration")
            
            model_config = self.model_configs[model_config_key]
            if not model_config.get("enabled", False):
                raise ValueError(f"Model {model_type.value} is not enabled in configuration")
            
            # Load the model
            logger.info(f"Loading model: {model_type.value}")
            model = await self._load_model(model_type, model_config)
            
            # Cache if enabled
            if self.cache_models:
                self.models[model_type] = model
            
            return model
    
    async def _load_model(self, model_type: ModelType, model_config: Dict) -> Any:
        """
        Load a specific model based on type
        
        Args:
            model_type: Type of model to load
            model_config: Configuration for the model
            
        Returns:
            Loaded model adapter
        """
        # Import adapters (lazy import to avoid circular dependencies)
        from .adapters import (
            BinaryClassifierAdapter,
            MultiClassClassifierAdapter,
            SegmentationAdapter,
            RAGAdapter,
            ReportGeneratorAdapter,
            FeatureExtractorAdapter
        )
        
        try:
            if model_type == ModelType.BINARY_CLASSIFIER:
                adapter = BinaryClassifierAdapter(
                    checkpoint_path=model_config.get("checkpoint_path"),
                    model_type=model_config.get("model_type", "swin_transformer"),
                    device=self.device
                )
            
            elif model_type == ModelType.MULTICLASS_CLASSIFIER:
                adapter = MultiClassClassifierAdapter(
                    checkpoint_path=model_config.get("checkpoint_path"),
                    num_classes=model_config.get("num_classes", 14),
                    model_type=model_config.get("model_type", "densenet121"),
                    device=self.device
                )
            
            elif model_type == ModelType.SEGMENTATION:
                adapter = SegmentationAdapter(
                    checkpoint_path=model_config.get("checkpoint_path"),
                    model_type=model_config.get("model_type", "unet"),
                    device=self.device
                )
            
            elif model_type == ModelType.RAG:
                adapter = RAGAdapter(
                    model_name=model_config.get("model_name"),
                    vector_db_path=model_config.get("vector_db_path"),
                    documents_path=model_config.get("documents_path"),
                    device=self.device
                )
            
            elif model_type == ModelType.REPORT_GENERATOR:
                adapter = ReportGeneratorAdapter(
                    model_name=model_config.get("model_name"),
                    checkpoint_path=model_config.get("checkpoint_path"),
                    device=self.device
                )
            
            elif model_type == ModelType.FEATURE_EXTRACTOR:
                adapter = FeatureExtractorAdapter(
                    model_type=model_config.get("model_type", "radiomics"),
                    device=self.device
                )
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            await adapter.load()
            logger.info(f"Successfully loaded model: {model_type.value}")
            return adapter
            
        except Exception as e:
            logger.error(f"Failed to load model {model_type.value}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    async def load_model(self, model_type: ModelType) -> None:
        """
        Explicitly load a model into the registry
        
        Args:
            model_type: Type of model to load
        """
        await self.get_model(model_type)
    
    async def unload_model(self, model_type: ModelType) -> None:
        """
        Unload a model from memory
        
        Args:
            model_type: Type of model to unload
        """
        async with self._locks[model_type]:
            if model_type in self.models:
                logger.info(f"Unloading model: {model_type.value}")
                
                # Call cleanup method if available
                model = self.models[model_type]
                if hasattr(model, 'cleanup'):
                    await model.cleanup()
                
                # Remove from registry
                del self.models[model_type]
                
                # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                logger.info(f"Model {model_type.value} unloaded successfully")
            else:
                logger.warning(f"Model {model_type.value} not loaded, nothing to unload")
    
    async def unload_all(self) -> None:
        """Unload all models from memory"""
        logger.info("Unloading all models...")
        for model_type in list(self.models.keys()):
            await self.unload_model(model_type)
        logger.info("All models unloaded")
    
    def list_models(self) -> Dict[str, Any]:
        """
        List all available models and their status
        
        Returns:
            Dictionary with model information
        """
        models_info = {}
        
        for model_type in ModelType:
            config_key = model_type.value
            if config_key in self.model_configs:
                model_config = self.model_configs[config_key]
                models_info[config_key] = {
                    "enabled": model_config.get("enabled", False),
                    "loaded": model_type in self.models,
                    "type": model_config.get("model_type", "unknown"),
                    "config": model_config
                }
        
        return {
            "device": str(self.device),
            "cache_enabled": self.cache_models,
            "loaded_models_count": len(self.models),
            "models": models_info
        }
    
    async def reload_model(self, model_type: ModelType) -> None:
        """
        Reload a model (useful after config changes)
        
        Args:
            model_type: Type of model to reload
        """
        logger.info(f"Reloading model: {model_type.value}")
        await self.unload_model(model_type)
        await self.load_model(model_type)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics
        
        Returns:
            Dictionary with memory information
        """
        memory_info = {
            "loaded_models": list(self.models.keys()),
            "loaded_count": len(self.models)
        }
        
        if torch.cuda.is_available():
            memory_info["cuda"] = {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
                "max_allocated": torch.cuda.max_memory_allocated() / 1024**3  # GB
            }
        
        return memory_info
