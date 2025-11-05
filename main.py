import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_streamlit_ui():
    """Run the Streamlit web interface"""
    import subprocess
    
    logger.info("🚀 Starting CXR Agent Streamlit UI...")
    app_path = Path(__file__).parent / "src" / "web" / "app.py"
    
    if not app_path.exists():
        logger.error(f"App file not found: {app_path}")
        sys.exit(1)
    
    # Run streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", "8501",
        "--server.address", "localhost"
    ])


def run_mcp_server():
    """Run the Model Context Protocol server"""
    logger.info("🚀 Starting CXR Agent MCP Server...")
    
    from src.servers.mcp_server import CXRMCPServer
    import asyncio
    
    async def start_mcp():
        config_path = Path(__file__).parent / "config" / "mcp_config.json"
        server = CXRMCPServer(config_path=str(config_path))
        await server.run()
    
    try:
        asyncio.run(start_mcp())
    except KeyboardInterrupt:
        logger.info("MCP Server stopped by user")


def run_cli_mode():
    """Run in interactive CLI mode"""
    import asyncio
    from src.agent import UnifiedAgent
    from src.rag.llm_engine import LLMEngine
    from src.rag.document_processor import VectorStore
    
    logger.info("🚀 Starting CXR Agent CLI Mode...")
    
    async def cli_loop():
        # Configuration
        config = {
            "models": {
                "binary_classifier": {
                    "enabled": True,
                    "checkpoint_path": "weights/binary_classifier.pth",
                    "model_type": "swin_transformer"
                },
                "multiclass_classifier": {
                    "enabled": True,
                    "checkpoint_path": "weights/fourteen_class_classifier",
                    "num_classes": 14,
                    "model_type": "swin_transformer"
                },
                "rag": {
                    "enabled": True,
                    "vector_db_path": "chroma_db",
                    "documents_path": "dataset/books"
                }
            },
            "device": "cuda"
        }
        
        print("\n" + "="*80)
        print("🫁 CXR AGENT - AI-Powered Chest X-ray Analysis")
        print("="*80 + "\n")
        
        print("Initializing agent components...")
        
        # Initialize
        try:
            vector_store = VectorStore(collection_name="respiratory_care_docs")
            
            # Determine model from environment
            import os
            model_name = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL")
            
            llm_engine = LLMEngine(
                model_name=model_name,
                max_tokens=2048,
                temperature=0.7
            )
            
            agent = UnifiedAgent(
                config=config,
                llm_engine=llm_engine,
                vector_store=vector_store
            )
            print("✅ Agent initialized!\n")
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            print(f"\n❌ Error: {e}")
            print("\nNote: Make sure you have:")
            print("  1. API keys set in .env file")
            print("  2. Weights in the weights/ directory")
            print("  3. RAG documents processed in chroma_db/")
            return
        
        print("Commands:")
        print("  - Type your question about chest X-rays or medical topics")
        print("  - Type 'upload <path>' to analyze an image")
        print("  - Type 'exit' or 'quit' to stop")
        print("  - Type 'help' for more information\n")
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\n📖 CXR Agent Help:")
                    print("  • Ask medical questions: 'What are signs of pneumonia?'")
                    print("  • Analyze images: 'upload path/to/xray.jpg'")
                    print("  • Combine both: Upload an image, then ask about it")
                    print("  • The agent remembers context within the session")
                    continue
                
                # Check for upload command
                image_path = None
                if user_input.lower().startswith('upload '):
                    image_path = user_input[7:].strip()
                    if not Path(image_path).exists():
                        print(f"❌ Error: Image not found: {image_path}")
                        continue
                    user_input = "Please analyze this chest X-ray image."
                
                # Process the message
                print("\n🤔 Thinking...")
                response = await agent.process_message(
                    query=user_input,
                    image_path=image_path
                )
                
                # Display response
                print("\n🤖 Agent:")
                print("-" * 80)
                print(response['answer'])
                print("-" * 80)
                
                # Show additional info if available
                if response.get('has_thinking') and response.get('thinking'):
                    print(f"\n🧠 Reasoning: {response['thinking'][:200]}...")
                
                if response.get('image_analysis'):
                    print("\n🔍 Image Analysis:")
                    analysis = response['image_analysis']
                    if analysis.get('binary'):
                        binary = analysis['binary']
                        print(f"  • Classification: {binary['prediction']} ({binary['confidence']:.1%})")
                    if analysis.get('diseases'):
                        diseases = analysis['diseases']
                        detected = [d for d in diseases if d['probability'] > 0.5]
                        if detected:
                            print(f"  • Diseases detected: {', '.join([d['disease'] for d in detected])}")
                
                if response.get('sources'):
                    print(f"\n📚 References: {len(response['sources'])} medical sources")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                print(f"\n❌ Error: {e}")
    
    try:
        asyncio.run(cli_loop())
    except KeyboardInterrupt:
        logger.info("\nCLI mode stopped by user")


def run_demo():
    """Run a quick demo with example queries"""
    import asyncio
    
    logger.info("🚀 Running CXR Agent Demo...")
    
    # Import and run the example
    sys.path.insert(0, str(Path(__file__).parent / "examples"))
    from examples.basic_usage import main as demo_main
    
    try:
        asyncio.run(demo_main())
    except KeyboardInterrupt:
        logger.info("\nDemo stopped by user")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CXR Agent - AI-Powered Chest X-ray Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Streamlit web interface (recommended for most users)
  python main.py --ui
  
  # Run in interactive CLI mode
  python main.py --cli
  
  # Run MCP server for integration
  python main.py --mcp
  
  # Run demo with examples
  python main.py --demo
        """
    )
    
    parser.add_argument(
        '--ui', '--streamlit',
        action='store_true',
        help='Run Streamlit web interface (default)'
    )
    parser.add_argument(
        '--cli', '--terminal',
        action='store_true',
        help='Run in interactive CLI mode'
    )
    parser.add_argument(
        '--mcp',
        action='store_true',
        help='Run Model Context Protocol server'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with example queries'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='CXR Agent v1.0.0'
    )
    
    args = parser.parse_args()
    
    # If no mode specified, default to UI
    if not any([args.ui, args.cli, args.mcp, args.demo]):
        args.ui = True
    
    try:
        if args.ui:
            run_streamlit_ui()
        elif args.cli:
            run_cli_mode()
        elif args.mcp:
            run_mcp_server()
        elif args.demo:
            run_demo()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
