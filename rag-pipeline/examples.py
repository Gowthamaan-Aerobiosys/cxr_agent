#!/usr/bin/env python3
"""
Example usage script for CXR Agent RAG Pipeline
Demonstrates various ways to use the system
"""

import os
import sys
import time
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_processor import DocumentProcessor, VectorStore
from qwen_agent import QwenAgent, AgenticRAG
from config import DEFAULT_CONFIG

def example_document_processing():
    """Example: Process PDF documents and create vector embeddings"""
    print("🔄 Example 1: Document Processing")
    print("-" * 40)
    
    # Initialize document processor
    processor = DocumentProcessor(
        chunk_size=DEFAULT_CONFIG.document.chunk_size,
        chunk_overlap=DEFAULT_CONFIG.document.chunk_overlap
    )
    
    # Process documents (example with small subset)
    dataset_path = "../dataset/books"
    if os.path.exists(dataset_path):
        print(f"Processing documents from: {dataset_path}")
        chunks = processor.process_documents(dataset_path)
        print(f"Generated {len(chunks)} text chunks")
        
        # Show sample chunk
        if chunks:
            sample_chunk = chunks[0]
            print(f"\nSample chunk:")
            print(f"Source: {sample_chunk.source}")
            print(f"Page: {sample_chunk.page_number}")
            print(f"Text preview: {sample_chunk.text[:200]}...")
    else:
        print(f"Dataset path not found: {dataset_path}")
    
    print("\n" + "="*60 + "\n")

def example_vector_store():
    """Example: Vector store operations"""
    print("🔍 Example 2: Vector Store Operations")
    print("-" * 40)
    
    # Initialize vector store
    vector_store = VectorStore(collection_name="example_collection")
    
    # Get collection statistics
    stats = vector_store.get_collection_stats()
    print(f"Collection: {stats['collection_name']}")
    print(f"Total documents: {stats['total_documents']}")
    
    if stats['total_documents'] > 0:
        # Example searches
        queries = [
            "mechanical ventilation",
            "PEEP settings",
            "respiratory failure",
            "ventilator weaning"
        ]
        
        print("\nExample searches:")
        for query in queries:
            print(f"\nQuery: '{query}'")
            results = vector_store.search(query, n_results=3)
            
            for i, result in enumerate(results, 1):
                relevance = 1 - result['distance']
                print(f"  {i}. {result['metadata']['source']} (Page {result['metadata']['page_number']}) - {relevance:.2%} relevant")
                print(f"     Preview: {result['text'][:100]}...")
    else:
        print("No documents in vector store. Run document processing first.")
    
    print("\n" + "="*60 + "\n")

def example_query_analysis():
    """Example: Query intent analysis"""
    print("🧠 Example 3: Query Intent Analysis")
    print("-" * 40)
    
    # Note: This creates a minimal agent for analysis only
    try:
        # Mock agent for demonstration (without loading full model)
        class MockAgent:
            def analyze_query_intent(self, query):
                from qwen_agent import QwenAgent
                agent = QwenAgent.__new__(QwenAgent)
                return agent.analyze_query_intent(query)
        
        agent = MockAgent()
        
        # Example queries
        queries = [
            "How do I set up mechanical ventilation for a COPD patient?",
            "What causes ARDS in patients?",
            "When should I use PEEP?",
            "What are normal ABG values?",
            "How to troubleshoot high pressure alarms?",
            "Emergency intubation procedure"
        ]
        
        print("Analyzing query intents:")
        for query in queries:
            print(f"\nQuery: '{query}'")
            analysis = agent.analyze_query_intent(query)
            
            print(f"  Type: {analysis['question_type']}")
            print(f"  Urgency: {analysis['urgency']}")
            
            detected_concepts = [k for k, v in analysis['concepts'].items() if v]
            if detected_concepts:
                print(f"  Concepts: {', '.join(detected_concepts)}")
    
    except Exception as e:
        print(f"Could not run query analysis: {e}")
        print("This requires the full QWEN model to be loaded.")
    
    print("\n" + "="*60 + "\n")

def example_quick_queries():
    """Example: Quick predefined queries"""
    print("⚡ Example 4: Quick Queries")
    print("-" * 40)
    
    from config import QUICK_QUERIES
    
    print("Available quick queries:")
    for i, query in enumerate(QUICK_QUERIES[:5], 1):  # Show first 5
        print(f"{i}. {query}")
    
    print(f"\n... and {len(QUICK_QUERIES) - 5} more queries available")
    print("\nThese can be used in the web interface or CLI for quick access.")
    
    print("\n" + "="*60 + "\n")

def example_configuration():
    """Example: Configuration management"""
    print("⚙️ Example 5: Configuration Management")
    print("-" * 40)
    
    from config import DEFAULT_CONFIG, save_config_to_file
    
    print("Current configuration:")
    print(f"Model: {DEFAULT_CONFIG.model.model_name}")
    print(f"Chunk size: {DEFAULT_CONFIG.document.chunk_size}")
    print(f"Collection: {DEFAULT_CONFIG.vector_store.collection_name}")
    print(f"Dataset path: {DEFAULT_CONFIG.dataset_path}")
    
    # Save configuration example
    config_file = "example_config.json"
    save_config_to_file(DEFAULT_CONFIG, config_file)
    print(f"\nConfiguration saved to: {config_file}")
    
    # Show environment variables
    print("\nEnvironment variables that can be set:")
    env_vars = [
        "QWEN_MODEL_NAME", "CHUNK_SIZE", "COLLECTION_NAME", 
        "DATASET_PATH", "LOG_LEVEL"
    ]
    for var in env_vars:
        value = os.getenv(var, "Not set")
        print(f"  {var}: {value}")
    
    print("\n" + "="*60 + "\n")

def example_full_pipeline():
    """Example: Full RAG pipeline (requires model loading)"""
    print("🚀 Example 6: Full RAG Pipeline")
    print("-" * 40)
    
    print("This example requires loading the QWEN model and processing documents.")
    print("It may take several minutes and requires significant computational resources.")
    
    proceed = input("Do you want to proceed? (y/N): ").lower().strip()
    
    if proceed == 'y':
        try:
            print("Initializing components...")
            
            # Initialize components
            processor = DocumentProcessor()
            vector_store = VectorStore()
            
            # Check if documents are already processed
            stats = vector_store.get_collection_stats()
            if stats['total_documents'] == 0:
                print("Processing documents (this may take a while)...")
                chunks = processor.process_documents("../dataset/books")
                if chunks:
                    vector_store.add_documents(chunks)
                    print(f"Processed {len(chunks)} document chunks")
                else:
                    print("No documents found to process")
                    return
            
            print("Loading QWEN model...")
            qwen_agent = QwenAgent()
            
            print("Initializing RAG system...")
            rag_system = AgenticRAG(vector_store, qwen_agent)
            
            # Example query
            query = "What are the key considerations for ventilator weaning?"
            print(f"\nProcessing query: '{query}'")
            
            response = rag_system.process_query(query)
            
            print("\nResponse:")
            print(response['answer'])
            
            print(f"\nQuery type: {response['query_type']}")
            print(f"Urgency: {response['urgency']}")
            print(f"Sources used: {len(response['sources'])}")
            
        except Exception as e:
            print(f"Error running full pipeline: {e}")
            print("This requires proper GPU setup and model access.")
    else:
        print("Skipping full pipeline example.")
    
    print("\n" + "="*60 + "\n")

def main():
    """Run all examples"""
    print("🫁 CXR Agent RAG Pipeline - Usage Examples")
    print("=" * 60)
    print()
    
    examples = [
        example_document_processing,
        example_vector_store,
        example_query_analysis,
        example_quick_queries,
        example_configuration,
        example_full_pipeline
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except KeyboardInterrupt:
            print("\nExamples interrupted by user.")
            break
        except Exception as e:
            print(f"Error in example {i}: {e}")
            continue
    
    print("Examples completed!")
    print("\nNext steps:")
    print("1. Run 'python main.py' for interactive CLI")
    print("2. Run 'python main.py --gui' for web interface")
    print("3. Check README.md for detailed documentation")

if __name__ == "__main__":
    main()
