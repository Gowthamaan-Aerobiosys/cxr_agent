#!/usr/bin/env python3
"""
CXR Agent Setup Script
Ensures all dependencies are installed and the system is ready to run
"""

import subprocess
import sys
import os
from pathlib import Path
import importlib

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported")
        print("📋 Requirements: Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing requirements...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully!")
            return True
        else:
            print(f"❌ Failed to install requirements:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_key_imports():
    """Check if key packages can be imported"""
    print("\n🔍 Checking key imports...")
    
    key_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("streamlit", "Streamlit"),
        ("fastapi", "FastAPI"),
        ("langchain", "LangChain"),
        ("chromadb", "ChromaDB"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas")
    ]
    
    success_count = 0
    
    for package, name in key_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {name}")
            success_count += 1
        except ImportError:
            print(f"  ❌ {name} - not available")
    
    print(f"\n📊 Import check: {success_count}/{len(key_packages)} packages available")
    return success_count == len(key_packages)

def setup_rag_pipeline():
    """Setup RAG pipeline if needed"""
    print("\n🔧 Setting up RAG pipeline...")
    
    rag_dir = Path(__file__).parent / "rag-pipeline"
    if not rag_dir.exists():
        print("❌ RAG pipeline directory not found!")
        return False
    
    # Check if setup script exists
    setup_script = rag_dir / "setup.bat"
    if setup_script.exists():
        print("📋 Found RAG setup script")
        print("💡 You may need to run 'cd rag-pipeline && setup.bat' manually")
    
    return True

def check_system_readiness():
    """Check if the system is ready to run"""
    print("\n🎯 Checking system readiness...")
    
    # Check main files exist
    main_files = [
        "cxr_agent.py",
        "chat_interface.py", 
        "mcp_server.py",
        "run_chat.py",
        "demo.py"
    ]
    
    missing_files = []
    for file in main_files:
        if not (Path(__file__).parent / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All main files present")
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*50)
    print("🚀 SETUP COMPLETE - NEXT STEPS")
    print("="*50)
    print("\n1. 📋 **Run Demo** (recommended first):")
    print("   python demo.py")
    print("   - Shows all system capabilities")
    print("   - Tests RAG queries and analysis workflow")
    
    print("\n2. 💬 **Start Chat Interface**:")
    print("   python run_chat.py")
    print("   - Opens interactive Streamlit interface")
    print("   - Upload images and ask questions")
    print("   - Available at: http://localhost:8501")
    
    print("\n3. 🖥️  **Use Command Line**:")
    print("   python cxr_cli.py --help")
    print("   - CLI for batch processing")
    print("   - Scriptable analysis workflows")
    
    print("\n4. 🌐 **Start MCP Server**:")
    print("   python mcp_server.py")
    print("   - API server for integration")
    print("   - Available at: http://localhost:8000")
    
    print("\n5. 🔧 **Setup RAG Pipeline** (if needed):")
    print("   cd rag-pipeline")
    print("   setup.bat")
    print("   - Downloads models and sets up knowledge base")
    
    print("\n" + "="*50)
    print("📚 Documentation: README.md")
    print("🐛 Issues: Check logs and error messages")
    print("="*50)

def main():
    """Main setup function"""
    print("🫁 CXR AGENT SETUP")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        print("\n💡 Try installing manually: pip install -r requirements.txt")
        return False
    
    # Check imports
    if not check_key_imports():
        print("\n💡 Some packages failed to import. You may need to install them manually.")
    
    # Setup RAG pipeline
    setup_rag_pipeline()
    
    # Check system readiness
    if not check_system_readiness():
        return False
    
    print("\n✅ Setup completed successfully!")
    print_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
