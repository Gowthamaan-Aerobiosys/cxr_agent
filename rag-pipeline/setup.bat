@echo off
REM CXR Agent RAG Pipeline Setup Script for Windows
REM This script sets up the environment and installs dependencies

echo 🫁 CXR Agent - Agentic RAG Pipeline Setup
echo =========================================

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is required but not found. Please install Python 3.8 or later.
    pause
    exit /b 1
)

python --version
echo ✅ Python detected

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created successfully
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📥 Installing dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully

REM Check if dataset exists
if exist "..\dataset\books" (
    echo ✅ Dataset directory found: ..\dataset\books
    for /f %%i in ('dir /b "..\dataset\books\*.pdf" 2^>nul ^| find /c /v ""') do set file_count=%%i
    echo 📚 Found %file_count% PDF files
) else (
    echo ⚠️ Dataset directory not found: ..\dataset\books
    echo    Please ensure your medical literature PDFs are in the correct location.
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "chroma_db" mkdir chroma_db
if not exist "logs" mkdir logs
if not exist "tests\__pycache__" mkdir tests\__pycache__

REM Copy environment template if .env doesn't exist
if not exist ".env" (
    echo 📝 Creating environment configuration...
    copy .env.example .env >nul
    echo ✅ Environment file created. You can edit .env to customize settings.
) else (
    echo ✅ Environment file already exists
)

REM Test imports
echo 🧪 Testing imports...
python -c "import torch; import transformers; import sentence_transformers; import chromadb; import streamlit; print('✅ All core libraries imported successfully')"

if %errorlevel% neq 0 (
    echo ❌ Import test failed. Please check the installation.
    pause
    exit /b 1
)

echo ✅ Import test passed

REM Check GPU availability
echo 🔍 Checking GPU availability...
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '⚠️ CUDA not available. Will use CPU (slower but functional)')"

echo.
echo 🎉 Setup completed successfully!
echo.
echo Next steps:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Run the system: python main.py
echo 3. Or start the web interface: python main.py --gui
echo.
echo For help: python main.py --help
pause
