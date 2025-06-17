#!/bin/bash

# CXR Agent RAG Pipeline Setup Script
# This script sets up the environment and installs dependencies

echo "🫁 CXR Agent - Agentic RAG Pipeline Setup"
echo "========================================="

# Check Python version
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Python detected: $python_version"
else
    echo "❌ Python 3 is required but not found. Please install Python 3.8 or later."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [[ $? -eq 0 ]]; then
        echo "✅ Virtual environment created successfully"
    else
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

if [[ $? -eq 0 ]]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check if dataset exists
if [ -d "../dataset/books" ]; then
    echo "✅ Dataset directory found: ../dataset/books"
    file_count=$(find ../dataset/books -name "*.pdf" | wc -l)
    echo "📚 Found $file_count PDF files"
else
    echo "⚠️ Dataset directory not found: ../dataset/books"
    echo "   Please ensure your medical literature PDFs are in the correct location."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p chroma_db
mkdir -p logs
mkdir -p tests/__pycache__

# Copy environment template if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating environment configuration..."
    cp .env.example .env
    echo "✅ Environment file created. You can edit .env to customize settings."
else
    echo "✅ Environment file already exists"
fi

# Test imports
echo "🧪 Testing imports..."
python3 -c "
import torch
import transformers
import sentence_transformers
import chromadb
import streamlit
print('✅ All core libraries imported successfully')
"

if [[ $? -eq 0 ]]; then
    echo "✅ Import test passed"
else
    echo "❌ Import test failed. Please check the installation."
    exit 1
fi

# Check GPU availability
echo "🔍 Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️ CUDA not available. Will use CPU (slower but functional)')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the system: python main.py"
echo "3. Or start the web interface: python main.py --gui"
echo ""
echo "For help: python main.py --help"
