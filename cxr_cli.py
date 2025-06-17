"""
CXR Agent CLI
Command-line interface for the CXR Agent system
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from cxr_agent import CXRAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CXRAgentCLI:
    """Command-line interface for CXR Agent"""
    
    def __init__(self):
        self.cxr_agent = None
    
    def initialize_agent(self, config_path: Optional[str] = None):
        """Initialize the CXR Agent"""
        logger.info("Initializing CXR Agent...")
        
        config = None
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                return False
        
        try:
            self.cxr_agent = CXRAgent(config)
            logger.info("CXR Agent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CXR Agent: {str(e)}")
            return False
    
    async def analyze_image(self, image_path: str, output_dir: str, 
                          include_rag: bool = True, generate_report: bool = True):
        """Analyze a single CXR image"""
        if not self.cxr_agent:
            logger.error("CXR Agent not initialized")
            return False
        
        try:
            logger.info(f"Analyzing image: {image_path}")
            
            # Perform analysis
            results = await self.cxr_agent.analyze_cxr(
                image_path=image_path,
                include_rag=include_rag,
                generate_report=generate_report
            )
            
            # Save results
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.cxr_agent.save_analysis_results(results, output_path)
            
            logger.info(f"Analysis completed. Results saved to {output_path}")
            
            # Print summary
            self._print_analysis_summary(results)
            
            return True
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return False
    
    async def batch_analyze(self, image_dir: str, output_dir: str,
                          include_rag: bool = True, generate_report: bool = True):
        """Analyze multiple CXR images in a directory"""
        if not self.cxr_agent:
            logger.error("CXR Agent not initialized")
            return False
        
        try:
            image_dir = Path(image_dir)
            if not image_dir.exists():
                logger.error(f"Image directory not found: {image_dir}")
                return False
            
            # Find image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(image_dir.glob(f"*{ext}"))
                image_files.extend(image_dir.glob(f"*{ext.upper()}"))
            
            if not image_files:
                logger.error(f"No image files found in {image_dir}")
                return False
            
            logger.info(f"Found {len(image_files)} images to analyze")
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Analyze each image
            for i, image_file in enumerate(image_files, 1):
                logger.info(f"Analyzing image {i}/{len(image_files)}: {image_file.name}")
                
                try:
                    results = await self.cxr_agent.analyze_cxr(
                        image_path=str(image_file),
                        include_rag=include_rag,
                        generate_report=generate_report
                    )
                    
                    # Save results in subdirectory
                    image_output_dir = output_path / image_file.stem
                    self.cxr_agent.save_analysis_results(results, image_output_dir)
                    
                    logger.info(f"Analysis {i} completed")
                    
                except Exception as e:
                    logger.error(f"Failed to analyze {image_file.name}: {str(e)}")
                    continue
            
            logger.info(f"Batch analysis completed. Results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {str(e)}")
            return False
    
    async def query_knowledge(self, question: str, context_file: Optional[str] = None):
        """Query medical knowledge using RAG"""
        if not self.cxr_agent:
            logger.error("CXR Agent not initialized")
            return False
        
        try:
            context = None
            if context_file:
                with open(context_file, 'r') as f:
                    context = json.load(f)
            
            logger.info(f"Querying: {question}")
            
            response = await self.cxr_agent.query_medical_knowledge(
                question=question,
                context=context
            )
            
            print("\\n" + "="*50)
            print("MEDICAL KNOWLEDGE QUERY RESPONSE")
            print("="*50)
            print(f"Question: {question}")
            print("\\nResponse:")
            print(response.get('response', 'No response available'))
            
            if 'sources' in response:
                print("\\nSources:")
                for source in response['sources']:
                    print(f"- {source}")
            
            print("="*50)
            
            return True
            
        except Exception as e:
            logger.error(f"Knowledge query failed: {str(e)}")
            return False
    
    def check_status(self):
        """Check system status"""
        if not self.cxr_agent:
            logger.error("CXR Agent not initialized")
            return False
        
        try:
            status = self.cxr_agent.get_system_status()
            
            print("\\n" + "="*50)
            print("CXR AGENT SYSTEM STATUS")
            print("="*50)
            
            print(f"Timestamp: {status['timestamp']}")
            print("\\nComponents:")
            for component, state in status['components'].items():
                print(f"  {component}: {state}")
            
            print("\\nConfiguration:")
            for key, value in status['config'].items():
                print(f"  {key}: {value}")
            
            if 'vector_store_stats' in status:
                print("\\nVector Store:")
                stats = status['vector_store_stats']
                if 'error' not in stats:
                    print(f"  Documents: {stats.get('total_documents', 'Unknown')}")
                    print(f"  Collections: {stats.get('collections', 'Unknown')}")
                else:
                    print(f"  Error: {stats['error']}")
            
            print("="*50)
            
            return True
            
        except Exception as e:
            logger.error(f"Status check failed: {str(e)}")
            return False
    
    def _print_analysis_summary(self, results):
        """Print analysis summary"""
        print("\\n" + "="*50)
        print("CXR ANALYSIS SUMMARY")
        print("="*50)
        
        # Classification results
        classification = results.get('classification', {})
        if classification:
            print("\\nClassification Results:")
            sorted_results = sorted(classification.items(), key=lambda x: x[1], reverse=True)
            for pathology, confidence in sorted_results[:5]:  # Top 5
                if confidence > 0.3:
                    print(f"  {pathology}: {confidence:.3f}")
        
        # Detected pathologies
        pathology_detection = results.get('pathology_detection', {})
        final_diagnosis = pathology_detection.get('final_diagnosis', {})
        
        if final_diagnosis:
            print("\\nDetected Pathologies:")
            for pathology, diagnosis in final_diagnosis.items():
                if 'High probability' in diagnosis or 'Moderate probability' in diagnosis:
                    print(f"  {pathology.replace('_', ' ').title()}: {diagnosis}")
        
        # RAG analysis
        rag_analysis = results.get('rag_analysis', {})
        if 'recommendations' in rag_analysis:
            print("\\nRecommendations:")
            for rec in rag_analysis['recommendations'][:3]:  # Top 3
                print(f"  - {rec}")
        
        print("="*50)

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="CXR Agent CLI")
    parser.add_argument("--config", help="Configuration file path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single CXR image")
    analyze_parser.add_argument("image_path", help="Path to CXR image")
    analyze_parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    analyze_parser.add_argument("--no-rag", action="store_true", help="Disable RAG analysis")
    analyze_parser.add_argument("--no-report", action="store_true", help="Disable report generation")
    
    # Batch analyze command
    batch_parser = subparsers.add_parser("batch", help="Analyze multiple CXR images")
    batch_parser.add_argument("image_dir", help="Directory containing CXR images")
    batch_parser.add_argument("--output-dir", "-o", required=True, help="Output directory")
    batch_parser.add_argument("--no-rag", action="store_true", help="Disable RAG analysis")
    batch_parser.add_argument("--no-report", action="store_true", help="Disable report generation")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query medical knowledge")
    query_parser.add_argument("question", help="Medical question to ask")
    query_parser.add_argument("--context", help="Context file (JSON)")
    
    # Status command
    subparsers.add_parser("status", help="Check system status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = CXRAgentCLI()
    
    if not cli.initialize_agent(args.config):
        sys.exit(1)
    
    # Execute command
    success = False
    
    if args.command == "analyze":
        success = asyncio.run(cli.analyze_image(
            image_path=args.image_path,
            output_dir=args.output_dir,
            include_rag=not args.no_rag,
            generate_report=not args.no_report
        ))
    
    elif args.command == "batch":
        success = asyncio.run(cli.batch_analyze(
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            include_rag=not args.no_rag,
            generate_report=not args.no_report
        ))
    
    elif args.command == "query":
        success = asyncio.run(cli.query_knowledge(
            question=args.question,
            context_file=args.context
        ))
    
    elif args.command == "status":
        success = cli.check_status()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
