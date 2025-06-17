#!/usr/bin/env python3
"""
CXR Agent Integration Script
Complete integration of all components for production deployment
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CXRAgentIntegrator:
    """Complete CXR Agent system integration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.components_status = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            "system": {
                "environment": "production",
                "log_level": "INFO",
                "max_workers": 4
            },
            "components": {
                "enable_rag": True,
                "enable_classification": True,
                "enable_segmentation": True,
                "enable_pathology_detection": True,
                "enable_mcp_server": True
            },
            "performance": {
                "batch_size": 8,
                "max_concurrent_requests": 10,
                "cache_enabled": True,
                "gpu_memory_fraction": 0.8
            }
        }
    
    async def initialize_system(self) -> bool:
        """Initialize the complete system"""
        logger.info("Initializing CXR Agent system...")
        
        try:
            # Step 1: Initialize core components
            if not await self._initialize_core_components():
                return False
            
            # Step 2: Initialize RAG system
            if self.config["components"]["enable_rag"]:
                if not await self._initialize_rag_system():
                    logger.warning("RAG system initialization failed, continuing without RAG")
                    self.config["components"]["enable_rag"] = False
            
            # Step 3: Initialize web services
            if self.config["components"]["enable_mcp_server"]:
                if not await self._initialize_mcp_server():
                    logger.warning("MCP server initialization failed")
                    return False
            
            # Step 4: Run system validation
            if not await self._validate_system():
                logger.error("System validation failed")
                return False
            
            logger.info("✓ CXR Agent system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}")
            return False
    
    async def _initialize_core_components(self) -> bool:
        """Initialize core image analysis components"""
        logger.info("Initializing core components...")
        
        try:
            # Test imports
            from lung_tools import (
                CXRImageProcessor,
                CXRClassifier,
                LungSegmenter,
                CXRFeatureExtractor,
                PathologyDetector
            )
            
            # Initialize components
            components = {}
            
            # Image processor
            components['image_processor'] = CXRImageProcessor()
            logger.info("✓ Image processor initialized")
            
            # Classifier
            if self.config["components"]["enable_classification"]:
                components['classifier'] = CXRClassifier()
                logger.info("✓ Classifier initialized")
            
            # Segmenter
            if self.config["components"]["enable_segmentation"]:
                components['segmenter'] = LungSegmenter()
                logger.info("✓ Segmenter initialized")
            
            # Feature extractor
            components['feature_extractor'] = CXRFeatureExtractor()
            logger.info("✓ Feature extractor initialized")
            
            # Pathology detector
            if self.config["components"]["enable_pathology_detection"]:
                components['pathology_detector'] = PathologyDetector()
                logger.info("✓ Pathology detector initialized")
            
            self.components_status['core'] = True
            return True
            
        except Exception as e:
            logger.error(f"Core components initialization failed: {str(e)}")
            self.components_status['core'] = False
            return False
    
    async def _initialize_rag_system(self) -> bool:
        """Initialize RAG system"""
        logger.info("Initializing RAG system...")
        
        try:
            # Check if RAG components are available
            rag_config_path = Path("rag-pipeline/config.py")
            if not rag_config_path.exists():
                logger.error("RAG pipeline configuration not found")
                return False
            
            # Import RAG components
            sys.path.append(str(Path("rag-pipeline").absolute()))
            
            from document_processor import DocumentProcessor, VectorStore
            from qwen_agent import QwenAgent, AgenticRAG
            from config import DEFAULT_CONFIG
            
            # Initialize document processor
            processor = DocumentProcessor(
                chunk_size=DEFAULT_CONFIG.document.chunk_size,
                chunk_overlap=DEFAULT_CONFIG.document.chunk_overlap
            )
            
            # Initialize vector store
            vector_store = VectorStore(
                collection_name=DEFAULT_CONFIG.vector_store.collection_name,
                embedding_model=DEFAULT_CONFIG.vector_store.embedding_model
            )
            
            # Check if documents are processed
            stats = vector_store.get_collection_stats()
            if stats['total_documents'] == 0:
                logger.warning("No documents found in vector store. Please run document processing first.")
                return False
            
            # Initialize QWEN agent  
            qwen_agent = QwenAgent(
                model_name=DEFAULT_CONFIG.model.model_name,
                load_in_4bit=DEFAULT_CONFIG.model.load_in_4bit,
                max_new_tokens=DEFAULT_CONFIG.model.max_new_tokens
            )
            
            # Create agentic RAG system
            rag_system = AgenticRAG(
                qwen_agent=qwen_agent,
                vector_store=vector_store,
                config=DEFAULT_CONFIG
            )
            
            self.components_status['rag'] = True
            logger.info("✓ RAG system initialized")
            return True
            
        except Exception as e:
            logger.error(f"RAG system initialization failed: {str(e)}")
            self.components_status['rag'] = False
            return False
    
    async def _initialize_mcp_server(self) -> bool:
        """Initialize MCP server"""
        logger.info("Initializing MCP server...")
        
        try:
            from mcp_server import CXRAgentMCPServer
            
            # Initialize server
            server = CXRAgentMCPServer(config=self.config)
            
            self.components_status['mcp_server'] = True
            logger.info("✓ MCP server initialized")
            return True
            
        except Exception as e:
            logger.error(f"MCP server initialization failed: {str(e)}")
            self.components_status['mcp_server'] = False
            return False
    
    async def _validate_system(self) -> bool:
        """Validate system functionality"""
        logger.info("Validating system...")
        
        try:
            # Import main CXR Agent
            from cxr_agent import CXRAgent
            
            # Initialize agent
            agent = CXRAgent(config=self.config)
            
            # Get system status
            status = agent.get_system_status()
            
            # Check critical components
            required_components = ['image_processor', 'classifier']
            for component in required_components:
                if status['components'].get(component) != 'operational':
                    logger.error(f"Critical component {component} not operational")
                    return False
            
            logger.info("✓ System validation passed")
            return True
            
        except Exception as e:
            logger.error(f"System validation failed: {str(e)}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            "configuration": self.config,
            "components_status": self.components_status,
            "system_paths": {
                "root": str(Path.cwd()),
                "config": self.config_path,
                "logs": "logs/",
                "uploads": "uploads/",
                "outputs": "outputs/"
            }
        }
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        logger.info("Running health check...")
        
        health_status = {
            "timestamp": str(asyncio.get_event_loop().time()),
            "overall_status": "healthy",
            "components": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            # Check core components
            from cxr_agent import CXRAgent
            agent = CXRAgent()
            status = agent.get_system_status()
            
            health_status["components"] = status["components"]
            
            # Check for issues
            for component, state in status["components"].items():
                if state != "operational":
                    health_status["warnings"].append(f"{component} is {state}")
            
            # Overall status
            if health_status["warnings"]:
                health_status["overall_status"] = "warning"
            
            if health_status["errors"]:
                health_status["overall_status"] = "error"
            
        except Exception as e:
            health_status["errors"].append(str(e))
            health_status["overall_status"] = "error"
        
        return health_status
    
    def print_system_summary(self):
        """Print system summary"""
        print("\n" + "="*60)
        print("CXR AGENT SYSTEM SUMMARY")
        print("="*60)
        
        print(f"Configuration: {self.config_path or 'Default'}")
        print(f"Environment: {self.config['system']['environment']}")
        
        print("\nComponents Status:")
        for component, status in self.components_status.items():
            status_text = "✓ Active" if status else "✗ Inactive"
            print(f"  {component}: {status_text}")
        
        print("\nEnabled Features:")
        for feature, enabled in self.config["components"].items():
            status_text = "✓" if enabled else "✗"
            print(f"  {feature}: {status_text}")
        
        print("\nUsage:")
        print("  CLI: python cxr_cli.py --help")
        print("  Server: python mcp_server.py")
        print("  Test: python test_system.py")
        
        print("="*60)

async def main():
    """Main integration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CXR Agent System Integration")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--health-check", action="store_true", help="Run health check only")
    parser.add_argument("--validate", action="store_true", help="Validate system only")
    
    args = parser.parse_args()
    
    # Initialize integrator
    integrator = CXRAgentIntegrator(config_path=args.config)
    
    if args.health_check:
        # Run health check
        health_status = await integrator.run_health_check()
        print(json.dumps(health_status, indent=2))
        return
    
    if args.validate:
        # Run validation only
        success = await integrator._validate_system()
        print(f"Validation: {'PASSED' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    
    # Full system initialization
    logger.info("Starting CXR Agent system integration...")
    
    success = await integrator.initialize_system()
    
    if success:
        integrator.print_system_summary()
        logger.info("System integration completed successfully!")
        
        # Save system info
        system_info = integrator.get_system_info()
        with open("system_info.json", 'w') as f:
            json.dump(system_info, f, indent=2)
        
        logger.info("System information saved to system_info.json")
        
    else:
        logger.error("System integration failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
