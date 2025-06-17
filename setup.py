#!/usr/bin/env python3
"""
Setup script for CXR Agent
Installs dependencies and sets up the environment
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a shell command with error handling"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    logger.info("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def check_cuda_availability():
    """Check CUDA availability for GPU acceleration"""
    logger.info("Checking CUDA availability...")
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("✓ CUDA is available")
            return True
        else:
            logger.info("CUDA not available, will use CPU")
            return False
    except Exception:
        logger.info("CUDA not available, will use CPU")
        return False

def create_directories():
    """Create necessary directories"""
    logger.info("Creating directories...")
    
    directories = [
        "uploads",
        "outputs", 
        "logs",
        "models",
        "data/processed",
        "data/raw",
        "configs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")
    
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    logger.info("Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    return True

def setup_rag_pipeline():
    """Setup RAG pipeline dependencies"""
    logger.info("Setting up RAG pipeline...")
    
    # Change to rag-pipeline directory and run setup
    if Path("rag-pipeline/setup.py").exists():
        if not run_command(f"cd rag-pipeline && {sys.executable} setup.py install", "Installing RAG pipeline"):
            return False
    
    # Run RAG setup script if it exists
    if Path("rag-pipeline/setup.sh").exists():
        if not run_command("cd rag-pipeline && bash setup.sh", "Running RAG setup script"):
            logger.warning("RAG setup script failed, continuing...")
    
    return True

def download_models():
    """Download required models (placeholder for actual model downloads)"""
    logger.info("Setting up models...")
    
    # This would download/setup required models
    # For now, just create model directories
    model_dirs = [
        "models/classification",
        "models/segmentation", 
        "models/embedding",
        "models/llm"
    ]
    
    for model_dir in model_dirs:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created model directory: {model_dir}")
    
    logger.info("Note: Models will be downloaded on first use")
    return True

def create_config_files():
    """Create default configuration files"""
    logger.info("Creating configuration files...")
    
    # Create default config
    config_content = """{
    "model": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "load_in_4bit": true,
        "max_new_tokens": 2048,
        "temperature": 0.7
    },
    "document": {
        "chunk_size": 1000,
        "chunk_overlap": 200
    },
    "vector_store": {
        "collection_name": "respiratory_care_docs",
        "embedding_model": "all-MiniLM-L6-v2",
        "persist_directory": "./chroma_db"
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1
    },
    "logging": {
        "level": "INFO",
        "file": "logs/cxr_agent.log"
    }
}"""
    
    config_path = Path("configs/default_config.json")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    logger.info(f"✓ Created configuration file: {config_path}")
    
    # Create environment file template
    env_content = """# CXR Agent Environment Variables
# Copy this to .env and configure as needed

# Model Configuration
QWEN_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
LOAD_IN_4BIT=true
MAX_NEW_TOKENS=2048

# Vector Store Configuration
COLLECTION_NAME=respiratory_care_docs
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DATASET_PATH=dataset/books

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_WORKERS=1

# Logging
LOG_LEVEL=INFO
DEBUG_MODE=false

# Optional: Hugging Face token for private models
# HUGGINGFACE_TOKEN=your_token_here

# Optional: OpenAI API key for comparisons
# OPENAI_API_KEY=your_key_here

# Optional: CUDA settings
# CUDA_VISIBLE_DEVICES=0
"""
    
    env_path = Path(".env.template")
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    logger.info(f"✓ Created environment template: {env_path}")
    return True

def setup_logging():
    """Setup logging configuration"""
    logger.info("Setting up logging...")
    
    log_config = """version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
  file:
    class: logging.FileHandler
    level: INFO
    formatter: default
    filename: logs/cxr_agent.log
    mode: a
loggers:
  cxr_agent:
    level: INFO
    handlers: [console, file]
    propagate: no
root:
  level: INFO
  handlers: [console]
"""
    
    log_config_path = Path("configs/logging.yaml")
    with open(log_config_path, 'w') as f:
        f.write(log_config)
    
    logger.info(f"✓ Created logging configuration: {log_config_path}")
    return True

def create_startup_scripts():
    """Create startup scripts"""
    logger.info("Creating startup scripts...")
    
    # Create Windows batch script
    batch_script = """@echo off
echo Starting CXR Agent...

REM Activate virtual environment if it exists
if exist "venv\\Scripts\\activate.bat" (
    call venv\\Scripts\\activate.bat
)

REM Start the MCP server
python mcp_server.py --host 0.0.0.0 --port 8000

pause
"""
    
    with open("start_server.bat", 'w') as f:
        f.write(batch_script)
    
    # Create Unix shell script
    shell_script = """#!/bin/bash
echo "Starting CXR Agent..."

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Start the MCP server
python mcp_server.py --host 0.0.0.0 --port 8000
"""
    
    with open("start_server.sh", 'w') as f:
        f.write(shell_script)
    
    # Make shell script executable
    if os.name != 'nt':  # Not Windows
        os.chmod("start_server.sh", 0o755)
    
    logger.info("✓ Created startup scripts")
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("CXR AGENT SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nNext steps:")
    print("1. Copy .env.template to .env and configure your settings")
    print("2. Place your medical documents in the 'dataset/books' directory")
    print("3. Run the setup for RAG pipeline:")
    print("   cd rag-pipeline && python setup.py")
    print("4. Start the system:")
    print("   - CLI: python cxr_cli.py --help")
    print("   - Server: python mcp_server.py")
    print("   - Or use: ./start_server.sh (Unix) or start_server.bat (Windows)")
    print("\nUsage examples:")
    print("- Analyze single image: python cxr_cli.py analyze image.jpg -o results/")
    print("- Batch analysis: python cxr_cli.py batch images/ -o results/")
    print("- Query knowledge: python cxr_cli.py query \"What is pneumonia?\"")
    print("- Check status: python cxr_cli.py status")
    print("\nAPI Server:")
    print("- Start server: python mcp_server.py")
    print("- Access at: http://localhost:8000")
    print("- API docs: http://localhost:8000/docs")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    logger.info("Starting CXR Agent setup...")
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Check CUDA
    cuda_available = check_cuda_availability()
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Installing Python dependencies", install_python_dependencies),
        ("Setting up RAG pipeline", setup_rag_pipeline),
        ("Setting up models", download_models),
        ("Creating configuration files", create_config_files),
        ("Setting up logging", setup_logging),
        ("Creating startup scripts", create_startup_scripts)
    ]
    
    failed_steps = []
    
    for description, function in steps:
        if not function():
            failed_steps.append(description)
    
    if failed_steps:
        logger.error(f"Setup completed with errors. Failed steps: {', '.join(failed_steps)}")
        logger.info("Please check the errors above and retry failed steps manually")
    else:
        logger.info("Setup completed successfully!")
        print_next_steps()

if __name__ == "__main__":
    main()
