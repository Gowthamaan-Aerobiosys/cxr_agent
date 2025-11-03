import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import numpy as np
from PIL import Image
import time

logger = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """
    Abstract base class for all model adapters
    
    All model adapters must implement:
    - load(): Load the model
    - predict() or appropriate inference method
    - cleanup(): Clean up resources
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.is_loaded = False
        self.load_time = None
    
    @abstractmethod
    async def load(self):
        """Load the model asynchronously"""
        pass
    
    @abstractmethod
    async def predict(self, *args, **kwargs):
        """Run inference"""
        pass
    
    async def cleanup(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BinaryClassifierAdapter(BaseModelAdapter):
    """
    Adapter for binary classification (Normal vs Abnormal)
    Wraps SwinCXR or similar binary classifier
    """
    
    def __init__(self, checkpoint_path: str, model_type: str = "swin_transformer", 
                 device: Optional[torch.device] = None):
        super().__init__(device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.labels = ["Normal", "Abnormal"]
    
    async def load(self):
        """Load the binary classification model"""
        start_time = time.time()
        
        try:
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
            
            # Extract model (handle different checkpoint formats)
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    self.model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    # Need to instantiate model first, then load state dict
                    self.model = self._create_model_architecture()
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    raise ValueError("Checkpoint format not recognized")
            else:
                self.model = checkpoint
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            logger.info(f"Binary classifier loaded in {self.load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load binary classifier: {e}")
            raise
    
    def _create_model_architecture(self):
        """Create model architecture based on model_type"""
        if self.model_type == "swin_transformer":
            import timm
            # Use timm Swin Large model for binary classification
            model = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=2)
            return model
        elif self.model_type == "resnet":
            from torchvision.models import resnet50
            model = resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            return model
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    async def predict(self, image_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict if CXR is normal or abnormal
        
        Args:
            image_path: Path to the CXR image
            threshold: Classification threshold
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Preprocess image
            image_tensor = await self._preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)
            
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            inference_time = time.time() - start_time
            
            # Interpret results
            predicted_class = "Abnormal" if probs[1] > threshold else "Normal"
            confidence = probs[1] if predicted_class == "Abnormal" else probs[0]
            
            return {
                "prediction": predicted_class,
                "confidence": float(confidence),
                "probabilities": {
                    "Normal": float(probs[0]),
                    "Abnormal": float(probs[1])
                },
                "threshold": threshold,
                "inference_time_ms": inference_time * 1000,
                "image_path": image_path
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    async def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input"""
        from torchvision import transforms
        
        # Standard ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor


class MultiClassClassifierAdapter(BaseModelAdapter):
    """
    Adapter for 14-class disease classification using Swin Transformer
    Detects multiple pathologies: Atelectasis, Cardiomegaly, etc.
    """
    
    def __init__(self, checkpoint_path: str, num_classes: int = 14,
                 model_type: str = "swin_transformer", device: Optional[torch.device] = None):
        super().__init__(device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.model_type = model_type
        
        # NIH ChestX-ray14 labels
        self.labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
    
    async def load(self):
        """Load the multi-class classification model"""
        start_time = time.time()
        
        try:
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    self.model = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    self.model = self._create_model_architecture()
                    self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model = checkpoint
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            logger.info(f"Multi-class classifier loaded in {self.load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load multi-class classifier: {e}")
            raise
    
    def _create_model_architecture(self):
        """Create Swin Transformer architecture for multi-label CXR classification"""
        import timm
        
        if self.model_type == "swin_transformer":
            # Use timm Swin Large model for multi-class classification
            model = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=self.num_classes)
            return model
        else:
            # Fallback to DenseNet if specified
            from torchvision.models import densenet121
            import torch.nn as nn
            model = densenet121(pretrained=False)
            num_features = model.classifier.in_features
            
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes),
                nn.Sigmoid()
            )
            return model
    
    async def predict(self, image_path: str, threshold: float = 0.3, 
                     top_k: int = 5) -> Dict[str, Any]:
        """
        Predict multiple diseases from CXR
        
        Args:
            image_path: Path to CXR image
            threshold: Minimum confidence threshold
            top_k: Return top K predictions
            
        Returns:
            Dictionary with disease predictions
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            # Preprocess
            image_tensor = await self._preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)
            
            # Inference
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probs = outputs.cpu().numpy()[0]  # Already sigmoid in model
            
            inference_time = time.time() - start_time
            
            # Create predictions dictionary
            predictions = {
                label: float(prob) 
                for label, prob in zip(self.labels, probs)
            }
            
            # Filter by threshold
            detected_diseases = {
                label: prob 
                for label, prob in predictions.items() 
                if prob >= threshold
            }
            
            # Sort by confidence
            sorted_predictions = sorted(
                predictions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_k]
            
            return {
                "all_predictions": predictions,
                "detected_diseases": detected_diseases,
                "top_predictions": dict(sorted_predictions),
                "num_detected": len(detected_diseases),
                "threshold": threshold,
                "inference_time_ms": inference_time * 1000,
                "image_path": image_path
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    async def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input"""
        from torchvision import transforms
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)


class SegmentationAdapter(BaseModelAdapter):
    """
    Adapter for lung segmentation
    Performs semantic segmentation of lung regions
    """
    
    def __init__(self, checkpoint_path: str, model_type: str = "unet",
                 device: Optional[torch.device] = None):
        super().__init__(device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
    
    async def load(self):
        """Load segmentation model"""
        start_time = time.time()
        
        try:
            # Import from existing lung_tools
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent / "lung_tools"))
            
            from lung_tools.segmentation import LungSegmenter
            
            self.model = LungSegmenter()
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            logger.info(f"Segmentation model loaded in {self.load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            raise
    
    async def predict(self, image_path: str, save_mask: bool = False,
                     output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Segment lung regions
        
        Args:
            image_path: Path to CXR image
            save_mask: Whether to save segmentation mask
            output_path: Path to save mask
            
        Returns:
            Segmentation results with metrics
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            start_time = time.time()
            
            # Run segmentation
            mask, metrics = self.model.segment_lungs(image_path)
            
            inference_time = time.time() - start_time
            
            # Save mask if requested
            if save_mask and output_path:
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                mask_image.save(output_path)
            
            return {
                "mask_shape": mask.shape,
                "lung_area_pixels": float(metrics.get("lung_area", 0)),
                "lung_ratio": float(metrics.get("lung_ratio", 0)),
                "left_lung_area": float(metrics.get("left_lung_area", 0)),
                "right_lung_area": float(metrics.get("right_lung_area", 0)),
                "inference_time_ms": inference_time * 1000,
                "mask_saved": save_mask and output_path is not None,
                "output_path": output_path if save_mask else None,
                "image_path": image_path
            }
            
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            raise


class RAGAdapter(BaseModelAdapter):
    """
    Adapter for RAG (Retrieval-Augmented Generation)
    Handles medical knowledge queries
    """
    
    def __init__(self, model_name: str, vector_db_path: str,
                 documents_path: str, device: Optional[torch.device] = None):
        super().__init__(device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_name = model_name
        self.vector_db_path = vector_db_path
        self.documents_path = documents_path
        self.agent = None
    
    async def load(self):
        """Load RAG agent"""
        start_time = time.time()
        
        try:
            from src.rag.llm_engine import AgenticRAG
            from src.rag.document_processor import VectorStore
            
            # Initialize vector store
            self.vector_store = VectorStore(persist_directory=self.vector_db_path)
            
            # Initialize agent
            self.agent = AgenticRAG(
                model_name=self.model_name,
                vector_store=self.vector_store
            )
            
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            logger.info(f"RAG agent loaded in {self.load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load RAG agent: {e}")
            raise
    
    async def predict(self, *args, **kwargs):
        """Alias for query method"""
        return await self.query(*args, **kwargs)
    
    async def query(self, query: str, top_k: int = 5, 
                   include_sources: bool = True) -> Dict[str, Any]:
        """
        Query medical knowledge base
        
        Args:
            query: Medical question
            top_k: Number of documents to retrieve
            include_sources: Include source references
            
        Returns:
            Response with answer and sources
        """
        if not self.is_loaded:
            raise RuntimeError("RAG agent not loaded. Call load() first.")
        
        try:
            start_time = time.time()
            
            # Run RAG query
            response = await self.agent.answer_question(
                query=query,
                top_k=top_k
            )
            
            query_time = time.time() - start_time
            
            result = {
                "query": query,
                "answer": response.get("answer", ""),
                "query_time_ms": query_time * 1000
            }
            
            if include_sources:
                result["sources"] = response.get("sources", [])
                result["retrieved_docs"] = response.get("num_docs", 0)
            
            return result
            
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            raise


class ReportGeneratorAdapter(BaseModelAdapter):
    """
    Adapter for radiology report generation
    Generates structured reports from CXR findings
    """
    
    def __init__(self, model_name: str, checkpoint_path: Optional[str] = None,
                 device: Optional[torch.device] = None):
        super().__init__(device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
    
    async def load(self):
        """Load report generation model"""
        start_time = time.time()
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None
            )
            
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            logger.info(f"Report generator loaded in {self.load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load report generator: {e}")
            raise
    
    async def predict(self, *args, **kwargs):
        """Alias for generate method"""
        return await self.generate(*args, **kwargs)
    
    async def generate(self, image_path: str, findings: Optional[Dict] = None,
                      clinical_context: str = "", style: str = "structured") -> Dict[str, Any]:
        """
        Generate radiology report
        
        Args:
            image_path: Path to CXR image
            findings: Structured findings from classification
            clinical_context: Patient information
            style: Report style (structured/narrative/brief)
            
        Returns:
            Generated report
        """
        if not self.is_loaded:
            raise RuntimeError("Report generator not loaded. Call load() first.")
        
        try:
            start_time = time.time()
            
            # Construct prompt
            prompt = self._build_prompt(findings, clinical_context, style)
            
            # Generate report
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            report = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            
            return {
                "report": report,
                "style": style,
                "findings_included": findings is not None,
                "clinical_context_included": bool(clinical_context),
                "generation_time_ms": generation_time * 1000,
                "image_path": image_path
            }
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            raise
    
    def _build_prompt(self, findings: Optional[Dict], clinical_context: str, 
                     style: str) -> str:
        """Build prompt for report generation"""
        prompt = "Generate a chest X-ray radiology report.\n\n"
        
        if clinical_context:
            prompt += f"Clinical Context:\n{clinical_context}\n\n"
        
        if findings:
            prompt += "Findings:\n"
            if "binary_classification" in findings:
                bc = findings["binary_classification"]
                prompt += f"- Overall: {bc.get('prediction', 'Unknown')}\n"
            
            if "disease_classification" in findings:
                dc = findings["disease_classification"]
                detected = dc.get("detected_diseases", {})
                if detected:
                    prompt += "- Detected pathologies:\n"
                    for disease, prob in detected.items():
                        prompt += f"  * {disease}: {prob:.1%}\n"
            
            prompt += "\n"
        
        prompt += f"Report Style: {style}\n\nReport:\n"
        
        return prompt


class FeatureExtractorAdapter(BaseModelAdapter):
    """
    Adapter for radiomics feature extraction
    Extracts quantitative features from CXR images
    """
    
    def __init__(self, model_type: str = "radiomics",
                 device: Optional[torch.device] = None):
        super().__init__(device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model_type = model_type
    
    async def load(self):
        """Load feature extractor"""
        start_time = time.time()
        
        try:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent / "lung_tools"))
            
            from lung_tools.feature_extractor import CXRFeatureExtractor
            
            self.model = CXRFeatureExtractor()
            self.is_loaded = True
            self.load_time = time.time() - start_time
            
            logger.info(f"Feature extractor loaded in {self.load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load feature extractor: {e}")
            raise
    
    async def predict(self, image_path: str, mask_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract features from CXR
        
        Args:
            image_path: Path to CXR image
            mask_path: Optional segmentation mask
            
        Returns:
            Extracted features
        """
        if not self.is_loaded:
            raise RuntimeError("Feature extractor not loaded. Call load() first.")
        
        try:
            start_time = time.time()
            
            features = self.model.extract_features(image_path, mask_path)
            
            extraction_time = time.time() - start_time
            
            return {
                "features": features,
                "num_features": len(features),
                "extraction_time_ms": extraction_time * 1000,
                "image_path": image_path
            }
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            raise
