#!/usr/bin/env python3
"""
Quick System Test
Rapid test of CXR Agent system readiness
"""

import sys
from pathlib import Path

def test_imports():
    """Test critical imports"""
    print("🔍 Testing imports...")
    
    critical_imports = [
        ("streamlit", "Streamlit UI framework"),
        ("fastapi", "FastAPI web framework"), 
        ("asyncio", "Async support"),
        ("pathlib", "Path handling"),
        ("json", "JSON processing"),
        ("datetime", "Date/time utilities")
    ]
    
    failed = []
    
    for module, description in critical_imports:
        try:
            __import__(module)
            print(f"  ✅ {description}")
        except ImportError:
            print(f"  ❌ {description} - MISSING")
            failed.append(module)
    
    return len(failed) == 0

def test_files():
    """Test if key files exist"""
    print("\n📁 Testing files...")
    
    key_files = [
        ("chat_interface.py", "Chat Interface"),
        ("cxr_agent.py", "CXR Agent Core"),
        ("mcp_server.py", "MCP Server"),
        ("run_chat.py", "Chat Launcher"),
        ("demo.py", "Demo Script"),
        ("requirements.txt", "Requirements"),
        ("lung_tools/__init__.py", "Lung Tools Package")
    ]
    
    missing = []
    
    for file_path, description in key_files:
        if Path(file_path).exists():
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ {description} - MISSING")
            missing.append(file_path)
    
    return len(missing) == 0

def test_chat_interface():
    """Test if chat interface can be imported"""
    print("\n💬 Testing chat interface...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path.cwd()))
        
        # Try to import (but don't run) the chat interface
        import chat_interface
        print("  ✅ Chat interface module loads successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Chat interface import failed: {e}")
        return False

def main():
    """Run quick system test"""
    print("🚀 CXR AGENT - QUICK SYSTEM TEST")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test files
    files_ok = test_files()
    
    # Test chat interface
    chat_ok = test_chat_interface()
    
    # Overall result
    print("\n" + "=" * 40)
    if imports_ok and files_ok and chat_ok:
        print("✅ SYSTEM TEST PASSED!")
        print("\n🚀 Ready to run:")
        print("  • python run_chat.py  (Start chat interface)")
        print("  • python demo.py      (Run full demo)")
        print("  • python mcp_server.py (Start API server)")
        return True
    else:
        print("❌ SYSTEM TEST FAILED!")
        print("\n🔧 Issues found:")
        if not imports_ok:
            print("  • Missing Python packages - run: pip install -r requirements.txt")
        if not files_ok:
            print("  • Missing system files - check file integrity")
        if not chat_ok:
            print("  • Chat interface issues - check dependencies")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)
