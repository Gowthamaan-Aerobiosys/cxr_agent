"""
Test script for CXR Agent system
Validates all components and functionality
"""

import asyncio
import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CXRAgentTester:
    """Comprehensive test suite for CXR Agent"""
    
    def __init__(self):
        self.test_results = {}
        self.server_url = "http://localhost:8000"
    
    def create_test_image(self) -> str:
        """Create a test CXR-like image"""
        # Create a synthetic CXR-like image
        img_array = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        
        # Add some structure to make it more CXR-like
        # Add darker regions for lungs
        img_array[100:400, 50:250] = img_array[100:400, 50:250] * 0.7  # Left lung
        img_array[100:400, 260:460] = img_array[100:400, 260:460] * 0.7  # Right lung
        
        # Save to temporary file
        temp_dir = Path("test_temp")
        temp_dir.mkdir(exist_ok=True)
        
        test_image_path = temp_dir / "test_cxr.png"
        Image.fromarray(img_array).save(test_image_path)
        
        return str(test_image_path)
    
    async def test_core_components(self) -> Dict[str, bool]:
        """Test core CXR Agent components"""
        results = {}
        
        try:
            logger.info("Testing core components...")
            
            # Test imports
            try:
                from cxr_agent import CXRAgent
                from lung_tools import CXRImageProcessor, CXRClassifier
                results['imports'] = True
                logger.info("✓ Core imports successful")
            except Exception as e:
                results['imports'] = False
                logger.error(f"✗ Import failed: {e}")
            
            # Test image processor
            try:
                from lung_tools import CXRImageProcessor
                processor = CXRImageProcessor()
                test_image_path = self.create_test_image()
                image = processor.load_image(test_image_path)
                tensor = processor.preprocess_for_model(image)
                results['image_processor'] = tensor is not None
                logger.info("✓ Image processor working")
            except Exception as e:
                results['image_processor'] = False
                logger.error(f"✗ Image processor failed: {e}")
            
            # Test classifier initialization
            try:
                from lung_tools import CXRClassifier
                classifier = CXRClassifier()
                results['classifier'] = True
                logger.info("✓ Classifier initialized")
            except Exception as e:
                results['classifier'] = False
                logger.error(f"✗ Classifier failed: {e}")
            
            # Test segmenter initialization
            try:
                from lung_tools import LungSegmenter
                segmenter = LungSegmenter()
                results['segmenter'] = True
                logger.info("✓ Segmenter initialized")
            except Exception as e:
                results['segmenter'] = False
                logger.error(f"✗ Segmenter failed: {e}")
        
        except Exception as e:
            logger.error(f"Core component test failed: {e}")
            results['general'] = False
        
        return results
    
    async def test_cxr_agent_integration(self) -> Dict[str, bool]:
        """Test CXR Agent integration"""
        results = {}
        
        try:
            logger.info("Testing CXR Agent integration...")
            
            # Initialize CXR Agent
            from cxr_agent import CXRAgent
            agent = CXRAgent()
            results['initialization'] = True
            logger.info("✓ CXR Agent initialized")
            
            # Test status check
            status = agent.get_system_status()
            results['status_check'] = 'components' in status
            logger.info("✓ Status check working")
            
            # Test image analysis (simplified)
            test_image_path = self.create_test_image()
            try:
                analysis_results = await agent.analyze_cxr(
                    image_path=test_image_path,
                    include_rag=False,  # Skip RAG for speed
                    generate_report=False
                )
                results['image_analysis'] = 'classification' in analysis_results
                logger.info("✓ Image analysis working")
            except Exception as e:
                results['image_analysis'] = False
                logger.error(f"✗ Image analysis failed: {e}")
        
        except Exception as e:
            logger.error(f"CXR Agent integration test failed: {e}")
            results['initialization'] = False
        
        return results
    
    def test_mcp_server(self) -> Dict[str, bool]:
        """Test MCP server endpoints"""
        results = {}
        
        try:
            logger.info("Testing MCP server...")
            
            # Test health endpoint
            try:
                response = requests.get(f"{self.server_url}/health", timeout=10)
                results['health_endpoint'] = response.status_code == 200
                logger.info("✓ Health endpoint working")
            except Exception as e:
                results['health_endpoint'] = False
                logger.error(f"✗ Health endpoint failed: {e}")
            
            # Test status endpoint
            try:
                response = requests.get(f"{self.server_url}/status", timeout=10)
                results['status_endpoint'] = response.status_code == 200
                logger.info("✓ Status endpoint working")
            except Exception as e:
                results['status_endpoint'] = False
                logger.error(f"✗ Status endpoint failed: {e}")
            
            # Test root endpoint
            try:
                response = requests.get(f"{self.server_url}/", timeout=10)
                results['root_endpoint'] = response.status_code == 200
                logger.info("✓ Root endpoint working")
            except Exception as e:
                results['root_endpoint'] = False
                logger.error(f"✗ Root endpoint failed: {e}")
        
        except Exception as e:
            logger.error(f"MCP server test failed: {e}")
            results['server_connection'] = False
        
        return results
    
    def test_file_structure(self) -> Dict[str, bool]:
        """Test required file structure"""
        results = {}
        
        logger.info("Testing file structure...")
        
        required_files = [
            "cxr_agent.py",
            "mcp_server.py", 
            "cxr_cli.py",
            "requirements.txt",
            "lung_tools/__init__.py",
            "lung_tools/image_processor.py",
            "lung_tools/classifier.py",
            "lung_tools/segmentation.py",
            "rag-pipeline/config.py",
            "rag-pipeline/qwen_agent.py"
        ]
        
        for file_path in required_files:
            exists = Path(file_path).exists()
            results[f"file_{file_path.replace('/', '_')}"] = exists
            if exists:
                logger.info(f"✓ {file_path} exists")
            else:
                logger.error(f"✗ {file_path} missing")
        
        return results
    
    def test_dependencies(self) -> Dict[str, bool]:
        """Test critical dependencies"""
        results = {}
        
        logger.info("Testing dependencies...")
        
        critical_packages = [
            'torch',
            'torchvision', 
            'transformers',
            'numpy',
            'opencv-python',
            'fastapi',
            'streamlit',
            'chromadb',
            'langchain'
        ]
        
        for package in critical_packages:
            try:
                __import__(package.replace('-', '_'))
                results[f"package_{package}"] = True
                logger.info(f"✓ {package} available")
            except ImportError:
                results[f"package_{package}"] = False
                logger.error(f"✗ {package} not available")
        
        return results
    
    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite"""
        logger.info("Starting CXR Agent test suite...")
        
        all_results = {}
        
        # Test file structure
        all_results['file_structure'] = self.test_file_structure()
        
        # Test dependencies
        all_results['dependencies'] = self.test_dependencies()
        
        # Test core components
        all_results['core_components'] = await self.test_core_components()
        
        # Test CXR Agent integration
        all_results['cxr_agent'] = await self.test_cxr_agent_integration()
        
        # Test MCP server (if running)
        all_results['mcp_server'] = self.test_mcp_server()
        
        return all_results
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate test report"""
        report_lines = []
        report_lines.append("CXR AGENT TEST REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        total_tests = 0
        passed_tests = 0
        
        for category, category_results in results.items():
            report_lines.append(f"{category.upper().replace('_', ' ')}:")
            
            for test_name, passed in category_results.items():
                status = "PASS" if passed else "FAIL"
                report_lines.append(f"  {test_name}: {status}")
                total_tests += 1
                if passed:
                    passed_tests += 1
            
            report_lines.append("")
        
        # Summary
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        report_lines.append("SUMMARY:")
        report_lines.append(f"  Total tests: {total_tests}")
        report_lines.append(f"  Passed: {passed_tests}")
        report_lines.append(f"  Failed: {total_tests - passed_tests}")
        report_lines.append(f"  Success rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            report_lines.append("  Overall status: EXCELLENT ✓")
        elif success_rate >= 70:
            report_lines.append("  Overall status: GOOD ✓")
        elif success_rate >= 50:
            report_lines.append("  Overall status: NEEDS IMPROVEMENT ⚠")
        else:
            report_lines.append("  Overall status: CRITICAL ISSUES ✗")
        
        return "\n".join(report_lines)
    
    def cleanup_test_files(self):
        """Clean up test files"""
        try:
            import shutil
            test_dir = Path("test_temp")
            if test_dir.exists():
                shutil.rmtree(test_dir)
                logger.info("✓ Test files cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up test files: {e}")

async def main():
    """Main test function"""
    tester = CXRAgentTester()
    
    try:
        # Run test suite
        results = await tester.run_full_test_suite()
        
        # Generate report
        report = tester.generate_test_report(results)
        
        # Print report
        print("\n" + report)
        
        # Save report to file
        report_path = Path("test_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Test report saved to {report_path}")
        
        # Save detailed results
        results_path = Path("test_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Detailed results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        tester.cleanup_test_files()

if __name__ == "__main__":
    asyncio.run(main())
