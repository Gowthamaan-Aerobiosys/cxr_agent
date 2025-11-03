"""
CXR Agent MCP Server
Model Context Protocol server for chest X-ray analysis
Provides a scalable, modular interface for multiple AI models
"""

import asyncio
import logging
from typing import Any, Dict, Optional
from pathlib import Path
import json
import traceback

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
    from mcp.server.stdio import stdio_server
except ImportError:
    raise ImportError(
        "MCP SDK not found. Install with: pip install mcp"
    )

from src.models.registry import ModelRegistry, ModelType
from src.models.adapters import (
    BinaryClassifierAdapter,
    MultiClassClassifierAdapter,
    SegmentationAdapter,
    RAGAdapter,
    ReportGeneratorAdapter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CXRMCPServer:
    """
    MCP Server for CXR Agent
    
    Provides a unified interface to multiple chest X-ray analysis models:
    - Binary classification (Normal/Abnormal)
    - 14-class disease classification
    - Lung segmentation
    - RAG-based medical Q&A
    - Report generation
    - Feature extraction
    
    The server is designed to be scalable and allows easy addition of new models.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the CXR MCP Server
        
        Args:
            config_path: Path to configuration file (YAML/JSON)
        """
        self.server = Server("cxr-agent")
        self.config_path = config_path or "config/mcp_config.json"
        self.config = self._load_config()
        
        # Initialize model registry
        self.model_registry = ModelRegistry(self.config)
        
        # Setup MCP tools
        self._setup_tools()
        
        logger.info("CXR MCP Server initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load server configuration"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                logger.warning(f"Config file not found at {self.config_path}, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "server_name": "cxr-agent",
            "version": "1.0.0",
            "models": {
                "binary_classifier": {
                    "enabled": True,
                    "checkpoint_path": "weights/binary_classifier.pth",
                    "model_type": "swin_transformer"
                },
                "multiclass_classifier": {
                    "enabled": True,
                    "checkpoint_path": "weights/fourteen_class_classifier",
                    "num_classes": 14,
                    "model_type": "densenet121"
                },
                "segmentation": {
                    "enabled": True,
                    "checkpoint_path": "weights/segmentation_model.pth",
                    "model_type": "unet"
                },
                "rag": {
                    "enabled": True,
                    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                    "vector_db_path": "rag-pipeline/chroma_db",
                    "documents_path": "dataset/books"
                },
                "report_generator": {
                    "enabled": False,
                    "model_name": "microsoft/BioGPT",
                    "checkpoint_path": "weights/report_generator.pth"
                }
            },
            "device": "cuda",  # "cuda" or "cpu"
            "cache_models": True,  # Keep models in memory
            "max_batch_size": 4
        }
    
    def _setup_tools(self):
        """Register all MCP tools"""
        
        # Tool 1: Binary Classification (Normal/Abnormal)
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List all available tools"""
            tools = []
            
            # Binary classification tool
            if self.config["models"]["binary_classifier"]["enabled"]:
                tools.append(Tool(
                    name="classify_cxr_binary",
                    description="Classify a chest X-ray image as Normal or Abnormal. "
                                "Returns probability scores for both classes.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Absolute or relative path to the chest X-ray image"
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Classification threshold (0-1), default 0.5",
                                "default": 0.5
                            }
                        },
                        "required": ["image_path"]
                    }
                ))
            
            # 14-class classification tool
            if self.config["models"]["multiclass_classifier"]["enabled"]:
                tools.append(Tool(
                    name="classify_cxr_diseases",
                    description="Detect 14 different pathologies in a chest X-ray: "
                                "Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, "
                                "Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, "
                                "Emphysema, Fibrosis, Pleural_Thickening, Hernia. "
                                "Returns probability scores for each disease.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Path to the chest X-ray image"
                            },
                            "threshold": {
                                "type": "number",
                                "description": "Minimum confidence threshold (0-1)",
                                "default": 0.3
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Return only top K predictions",
                                "default": 5
                            }
                        },
                        "required": ["image_path"]
                    }
                ))
            
            # Segmentation tool
            if self.config["models"]["segmentation"]["enabled"]:
                tools.append(Tool(
                    name="segment_lungs",
                    description="Perform semantic segmentation of lung regions in a chest X-ray. "
                                "Returns segmentation mask and lung area metrics.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Path to the chest X-ray image"
                            },
                            "save_mask": {
                                "type": "boolean",
                                "description": "Whether to save the segmentation mask",
                                "default": False
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Path to save the segmentation mask (if save_mask=True)"
                            }
                        },
                        "required": ["image_path"]
                    }
                ))
            
            # RAG tool for medical queries
            if self.config["models"]["rag"]["enabled"]:
                tools.append(Tool(
                    name="query_medical_knowledge",
                    description="Query the medical knowledge base about chest X-rays, "
                                "respiratory diseases, radiology interpretation, and clinical guidelines. "
                                "Uses RAG (Retrieval-Augmented Generation) with medical literature.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The medical question or query"
                            },
                            "top_k_docs": {
                                "type": "integer",
                                "description": "Number of relevant documents to retrieve",
                                "default": 5
                            },
                            "include_sources": {
                                "type": "boolean",
                                "description": "Include source references in the response",
                                "default": True
                            }
                        },
                        "required": ["query"]
                    }
                ))
            
            # Report generation tool
            if self.config["models"]["report_generator"]["enabled"]:
                tools.append(Tool(
                    name="generate_cxr_report",
                    description="Generate a comprehensive radiology report based on CXR findings. "
                                "Can integrate classification results and clinical context.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Path to the chest X-ray image"
                            },
                            "findings": {
                                "type": "object",
                                "description": "Structured findings from classification/segmentation"
                            },
                            "clinical_context": {
                                "type": "string",
                                "description": "Patient clinical information and history"
                            },
                            "report_style": {
                                "type": "string",
                                "enum": ["structured", "narrative", "brief"],
                                "default": "structured"
                            }
                        },
                        "required": ["image_path"]
                    }
                ))
            
            # Combined analysis tool
            tools.append(Tool(
                name="analyze_cxr_complete",
                description="Perform comprehensive chest X-ray analysis including: "
                            "binary classification, disease detection, lung segmentation, "
                            "and optional report generation. Returns all results combined.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to the chest X-ray image"
                        },
                        "include_segmentation": {
                            "type": "boolean",
                            "description": "Include lung segmentation",
                            "default": True
                        },
                        "include_report": {
                            "type": "boolean",
                            "description": "Generate radiology report",
                            "default": False
                        },
                        "clinical_context": {
                            "type": "string",
                            "description": "Patient clinical information"
                        }
                    },
                    "required": ["image_path"]
                }
            ))
            
            # Model management tools
            tools.extend([
                Tool(
                    name="list_available_models",
                    description="List all available models and their status (loaded/unloaded)",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="load_model",
                    description="Load a specific model into memory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "enum": ["binary_classifier", "multiclass_classifier", 
                                        "segmentation", "rag", "report_generator"]
                            }
                        },
                        "required": ["model_name"]
                    }
                ),
                Tool(
                    name="unload_model",
                    description="Unload a model from memory to free resources",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "enum": ["binary_classifier", "multiclass_classifier", 
                                        "segmentation", "rag", "report_generator"]
                            }
                        },
                        "required": ["model_name"]
                    }
                )
            ])
            
            return tools
        
        # Tool execution handler
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> list[TextContent]:
            """Execute tool based on name"""
            try:
                logger.info(f"Executing tool: {name} with arguments: {arguments}")
                
                if name == "classify_cxr_binary":
                    result = await self._classify_binary(arguments)
                elif name == "classify_cxr_diseases":
                    result = await self._classify_diseases(arguments)
                elif name == "segment_lungs":
                    result = await self._segment_lungs(arguments)
                elif name == "query_medical_knowledge":
                    result = await self._query_rag(arguments)
                elif name == "generate_cxr_report":
                    result = await self._generate_report(arguments)
                elif name == "analyze_cxr_complete":
                    result = await self._analyze_complete(arguments)
                elif name == "list_available_models":
                    result = await self._list_models()
                elif name == "load_model":
                    result = await self._load_model(arguments["model_name"])
                elif name == "unload_model":
                    result = await self._unload_model(arguments["model_name"])
                else:
                    result = {"error": f"Unknown tool: {name}"}
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
                
            except Exception as e:
                logger.error(f"Error executing tool {name}: {e}")
                logger.error(traceback.format_exc())
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }, indent=2)
                )]
    
    async def _classify_binary(self, args: Dict) -> Dict:
        """Binary classification: Normal vs Abnormal"""
        image_path = args["image_path"]
        threshold = args.get("threshold", 0.5)
        
        # Get or load model
        model = await self.model_registry.get_model(ModelType.BINARY_CLASSIFIER)
        
        # Run inference
        result = await model.predict(image_path, threshold=threshold)
        return result
    
    async def _classify_diseases(self, args: Dict) -> Dict:
        """Multi-class disease classification"""
        image_path = args["image_path"]
        threshold = args.get("threshold", 0.3)
        top_k = args.get("top_k", 5)
        
        model = await self.model_registry.get_model(ModelType.MULTICLASS_CLASSIFIER)
        result = await model.predict(image_path, threshold=threshold, top_k=top_k)
        return result
    
    async def _segment_lungs(self, args: Dict) -> Dict:
        """Lung segmentation"""
        image_path = args["image_path"]
        save_mask = args.get("save_mask", False)
        output_path = args.get("output_path")
        
        model = await self.model_registry.get_model(ModelType.SEGMENTATION)
        result = await model.predict(
            image_path, 
            save_mask=save_mask, 
            output_path=output_path
        )
        return result
    
    async def _query_rag(self, args: Dict) -> Dict:
        """RAG-based medical knowledge query"""
        query = args["query"]
        top_k_docs = args.get("top_k_docs", 5)
        include_sources = args.get("include_sources", True)
        
        model = await self.model_registry.get_model(ModelType.RAG)
        result = await model.query(
            query, 
            top_k=top_k_docs, 
            include_sources=include_sources
        )
        return result
    
    async def _generate_report(self, args: Dict) -> Dict:
        """Generate radiology report"""
        image_path = args["image_path"]
        findings = args.get("findings", {})
        clinical_context = args.get("clinical_context", "")
        report_style = args.get("report_style", "structured")
        
        model = await self.model_registry.get_model(ModelType.REPORT_GENERATOR)
        result = await model.generate(
            image_path, 
            findings=findings,
            clinical_context=clinical_context,
            style=report_style
        )
        return result
    
    async def _analyze_complete(self, args: Dict) -> Dict:
        """Comprehensive analysis combining all models"""
        image_path = args["image_path"]
        include_segmentation = args.get("include_segmentation", True)
        include_report = args.get("include_report", False)
        clinical_context = args.get("clinical_context", "")
        
        results = {
            "image_path": image_path,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Binary classification
        try:
            binary_model = await self.model_registry.get_model(ModelType.BINARY_CLASSIFIER)
            results["binary_classification"] = await binary_model.predict(image_path)
        except Exception as e:
            results["binary_classification"] = {"error": str(e)}
        
        # Disease classification
        try:
            disease_model = await self.model_registry.get_model(ModelType.MULTICLASS_CLASSIFIER)
            results["disease_classification"] = await disease_model.predict(image_path)
        except Exception as e:
            results["disease_classification"] = {"error": str(e)}
        
        # Segmentation
        if include_segmentation:
            try:
                seg_model = await self.model_registry.get_model(ModelType.SEGMENTATION)
                results["segmentation"] = await seg_model.predict(image_path)
            except Exception as e:
                results["segmentation"] = {"error": str(e)}
        
        # Report generation
        if include_report:
            try:
                report_model = await self.model_registry.get_model(ModelType.REPORT_GENERATOR)
                results["report"] = await report_model.generate(
                    image_path,
                    findings=results,
                    clinical_context=clinical_context
                )
            except Exception as e:
                results["report"] = {"error": str(e)}
        
        return results
    
    async def _list_models(self) -> Dict:
        """List all available models"""
        return self.model_registry.list_models()
    
    async def _load_model(self, model_name: str) -> Dict:
        """Load a specific model"""
        try:
            model_type = ModelType[model_name.upper()]
            await self.model_registry.load_model(model_type)
            return {
                "status": "success",
                "message": f"Model {model_name} loaded successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def _unload_model(self, model_name: str) -> Dict:
        """Unload a specific model"""
        try:
            model_type = ModelType[model_name.upper()]
            await self.model_registry.unload_model(model_type)
            return {
                "status": "success",
                "message": f"Model {model_name} unloaded successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting CXR MCP Server...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CXR Agent MCP Server")
    parser.add_argument(
        "--config",
        type=str,
        default="config/mcp_config.json",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    server = CXRMCPServer(config_path=args.config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
