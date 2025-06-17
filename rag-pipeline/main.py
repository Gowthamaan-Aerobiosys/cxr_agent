#!/usr/bin/env python3
"""
CXR Agent - Agentic RAG Pipeline Main Script
This script provides a command-line interface for the respiratory care AI system.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_processor import DocumentProcessor, VectorStore
from qwen_agent import QwenAgent, AgenticRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CXRAgentCLI:
    """Command-line interface for CXR Agent"""
    
    def __init__(self):
        self.rag_system = None
        self.vector_store = None
    
    def initialize_system(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """Initialize the RAG system"""
        logger.info("Initializing CXR Agent system...")
        
        try:
            # Initialize document processor
            processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
            
            # Initialize vector store
            self.vector_store = VectorStore(collection_name="respiratory_care_docs")
            
            # Check if we need to process documents
            stats = self.vector_store.get_collection_stats()
            logger.info(f"Vector store contains {stats['total_documents']} documents")
            
            if stats['total_documents'] == 0:
                logger.info("Vector store is empty. Processing documents...")
                self.process_documents(processor)
            
            # Initialize QWEN agent
            qwen_agent = QwenAgent(
                model_name=model_name,
                load_in_4bit=True,
                max_new_tokens=2048
            )
            
            # Initialize agentic RAG
            self.rag_system = AgenticRAG(self.vector_store, qwen_agent)
            
            logger.info("System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system: {str(e)}")
            return False
    
    def process_documents(self, processor: DocumentProcessor, 
                         dataset_path: str = "../dataset/books"):
        """Process documents and add to vector store"""
        logger.info(f"Processing documents from: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path not found: {dataset_path}")
            return False
        
        # Process documents
        chunks = processor.process_documents(dataset_path)
        
        if chunks:
            logger.info(f"Adding {len(chunks)} chunks to vector store...")
            self.vector_store.add_documents(chunks)
            logger.info("Documents processed successfully!")
            return True
        else:
            logger.warning("No document chunks generated")
            return False
    
    def interactive_mode(self):
        """Run interactive Q&A mode"""
        print("\n🫁 CXR Agent - Respiratory Care Assistant")
        print("=" * 50)
        print("Ask questions about mechanical ventilation, respiratory care, and pulmonary medicine.")
        print("Type 'quit', 'exit', or 'q' to end the session.")
        print("Type 'history' to see conversation history.")
        print("Type 'clear' to clear conversation history.")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👨‍⚕️ Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using CXR Agent. Stay safe! 🫁")
                    break
                
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                
                elif user_input.lower() == 'clear':
                    self.rag_system.clear_history()
                    print("✅ Conversation history cleared.")
                    continue
                
                elif not user_input:
                    print("Please enter a question.")
                    continue
                
                # Process query
                print("\n🔍 Analyzing your question...")
                response = self.rag_system.process_query(user_input)
                
                # Display response
                self.display_response(response)
                
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye! 🫁")
                break
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                print(f"❌ Error: {str(e)}")
    
    def display_response(self, response: Dict[str, Any]):
        """Display formatted response"""
        print(f"\n🤖 CXR Agent Response:")
        print("-" * 30)
        print(response['answer'])
        
        # Display metadata
        print(f"\n📊 Analysis:")
        print(f"   Question Type: {response['query_type'].title()}")
        print(f"   Urgency Level: {response['urgency'].title()}")
        
        # Display concepts
        concepts = response['concepts_detected']
        detected_concepts = [k for k, v in concepts.items() if v]
        if detected_concepts:
            print(f"   Concepts: {', '.join(detected_concepts).title()}")
        
        # Display sources
        if response['sources']:
            print(f"\n📚 Sources Referenced:")
            for i, source in enumerate(response['sources'], 1):
                relevance = source['relevance_score']
                print(f"   {i}. {source['source']} (Page {source['page']}) - {relevance:.1%} relevant")
    
    def show_history(self):
        """Display conversation history"""
        history = self.rag_system.get_conversation_history()
        
        if not history:
            print("📝 No conversation history yet.")
            return
        
        print(f"\n📝 Conversation History ({len(history)} queries):")
        print("=" * 50)
        
        for i, entry in enumerate(history, 1):
            print(f"\n{i}. Query: {entry['query']}")
            print(f"   Type: {entry['intent']['question_type']}")
            print(f"   Urgency: {entry['intent']['urgency']}")
            print(f"   Sources: {entry['sources_used']}")
            print(f"   Time: {entry['timestamp']}")
    
    def single_query(self, query: str):
        """Process single query and return response"""
        try:
            response = self.rag_system.process_query(query)
            self.display_response(response)
            return response
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            print(f"❌ Error: {str(e)}")
            return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="CXR Agent - Respiratory Care AI Assistant")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", 
                       help="QWEN model name to use")
    parser.add_argument("--query", type=str, help="Single query to process")
    parser.add_argument("--dataset", default="../dataset/books", 
                       help="Path to dataset directory")
    parser.add_argument("--reprocess", action="store_true", 
                       help="Force reprocessing of documents")
    parser.add_argument("--gui", action="store_true",
                       help="Launch Streamlit GUI")
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = CXRAgentCLI()
    
    # Initialize system
    if not cli.initialize_system(args.model):
        logger.error("Failed to initialize system")
        return 1
    
    # Reprocess documents if requested
    if args.reprocess:
        processor = DocumentProcessor()
        cli.process_documents(processor, args.dataset)
    
    # Launch GUI mode
    if args.gui:
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        return 0
    
    # Single query mode
    if args.query:
        cli.single_query(args.query)
        return 0
    
    # Interactive mode
    cli.interactive_mode()
    return 0

if __name__ == "__main__":
    sys.exit(main())
