#!/usr/bin/env python3
"""
CXR Agent Chat Launcher
Launches the Streamlit-based chat interface for CXR analysis and RAG queries
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the CXR Agent chat interface"""
    
    # Get the current directory (where this script is located)
    current_dir = Path(__file__).parent
    chat_interface_path = current_dir / "chat_interface.py"
    
    # Check if chat_interface.py exists
    if not chat_interface_path.exists():
        print("❌ Error: chat_interface.py not found!")
        print(f"Expected path: {chat_interface_path}")
        sys.exit(1)
    
    # Change to the project directory
    os.chdir(current_dir)
    
    print("🚀 Starting CXR Agent Chat Interface...")
    print(f"📁 Working directory: {current_dir}")
    print("🌐 Opening in browser at: http://localhost:8501")
    print("\n" + "="*50)
    print("CXR Agent Chat Features:")
    print("• 🔍 Analyze chest X-rays")
    print("• 💬 Ask medical questions")
    print("• 📊 Generate clinical reports")
    print("• 🧠 Get evidence-based interpretations")
    print("="*50 + "\n")
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, 
            "-m", "streamlit", "run", 
            str(chat_interface_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check if Streamlit is installed: pip install streamlit")
        print("3. Try running manually: streamlit run chat_interface.py")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Chat interface stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
