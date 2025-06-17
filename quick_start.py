#!/usr/bin/env python3
"""
CXR Agent Quick Start
Get the system running quickly with minimal setup
"""

import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, description, timeout=300):
    """Run command with timeout"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        logger.info(f"✓ {description} completed")
        return True, result.stdout
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {description} timed out")
        return False, "Timeout"
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed: {e.stderr}")
        return False, e.stderr

async def quick_start():
    """Quick start sequence"""
    print("🚀 CXR Agent Quick Start")
    print("=" * 40)
    
    # Step 1: Check Python version
    print("1. Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    
    # Step 2: Install dependencies
    print("\n2. Installing dependencies...")
    success, output = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages",
        timeout=600
    )
    if not success:
        print("❌ Dependency installation failed")
        print("Try: pip install torch torchvision transformers fastapi streamlit")
        return False
    
    # Step 3: Create directories
    print("\n3. Creating directories...")
    directories = ["uploads", "outputs", "logs", "configs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")
    
    # Step 4: Test imports
    print("\n4. Testing core imports...")
    try:
        import torch
        import transformers
        import fastapi
        import streamlit
        print("✅ Core packages available")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Step 5: Initialize system
    print("\n5. Initializing system...")
    try:
        success, output = run_command(
            f"{sys.executable} integrate_system.py --validate",
            "System validation",
            timeout=120
        )
        if success:
            print("✅ System validation passed")
        else:
            print("⚠️ System validation had issues, but continuing...")
    except Exception as e:
        print(f"⚠️ Validation error: {e}")
    
    # Step 6: Start services
    print("\n6. Starting services...")
    
    # Check if we can start the server
    try:
        import uvicorn
        print("✅ Ready to start MCP server")
        
        print("\n🎉 Quick start completed!")
        print("\nNext steps:")
        print("1. Start the server: python mcp_server.py")
        print("2. Or use CLI: python cxr_cli.py --help")
        print("3. Run tests: python test_system.py")
        print("4. Access API docs: http://localhost:8000/docs")
        
        # Ask if user wants to start server now
        response = input("\nStart MCP server now? (y/n): ").lower().strip()
        if response == 'y':
            print("\nStarting MCP server...")
            print("Access the API at: http://localhost:8000")
            print("Press Ctrl+C to stop")
            
            # Start server
            from mcp_server import main
            main()
        
        return True
        
    except ImportError:
        print("❌ uvicorn not available for server")
        return False

def main():
    """Main function"""
    try:
        success = asyncio.run(quick_start())
        if not success:
            print("\n❌ Quick start failed")
            print("\nTroubleshooting:")
            print("1. Check Python version (3.8+ required)")
            print("2. Install dependencies: pip install -r requirements.txt")
            print("3. Check CUDA availability for GPU support")
            print("4. Review logs for specific errors")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Quick start interrupted")
    except Exception as e:
        print(f"\n❌ Quick start error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
