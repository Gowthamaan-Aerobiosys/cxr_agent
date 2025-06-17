#!/usr/bin/env python3
"""
Demo script for CXR Agent RAG Pipeline
Shows a simple example of how to use the system without full model loading
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_system_overview():
    """Demonstrate system components without loading heavy models"""
    print("🫁 CXR Agent - Agentic RAG Pipeline Demo")
    print("=" * 60)
    
    # 1. Configuration Demo
    print("\n1. 📋 Configuration System")
    print("-" * 30)
    
    from config import DEFAULT_CONFIG, QUICK_QUERIES, MEDICAL_CONCEPTS
    
    print(f"Default Model: {DEFAULT_CONFIG.model.model_name}")
    print(f"Chunk Size: {DEFAULT_CONFIG.document.chunk_size}")
    print(f"Vector Store: {DEFAULT_CONFIG.vector_store.collection_name}")
    
    print(f"\nQuick Queries Available: {len(QUICK_QUERIES)}")
    print("Sample queries:")
    for i, query in enumerate(QUICK_QUERIES[:3], 1):
        print(f"  {i}. {query}")
    
    print(f"\nMedical Concept Categories: {list(MEDICAL_CONCEPTS.keys())}")
    
    # 2. Document Processing Demo
    print("\n2. 📄 Document Processing")
    print("-" * 30)
    
    from document_processor import DocumentProcessor
    
    processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
    
    # Demo text processing
    sample_text = """
    Mechanical ventilation is a life-support method used in critically ill patients with respiratory failure. 
    The primary goal is to maintain adequate gas exchange while minimizing ventilator-induced lung injury. 
    Key parameters include tidal volume, respiratory rate, PEEP, and FiO2. 
    Lung-protective ventilation strategies are essential for patients with ARDS.
    """
    
    metadata = {'source': 'demo.pdf', 'page_number': 1}
    chunks = processor.chunk_text(sample_text, metadata)
    
    print(f"Sample text processed into {len(chunks)} chunks")
    if chunks:
        print(f"First chunk preview: {chunks[0].text[:100]}...")
    
    # 3. Vector Store Demo
    print("\n3. 🔍 Vector Store Setup")
    print("-" * 30)
    
    from document_processor import VectorStore
    
    # Initialize vector store (this will create the collection)
    try:
        vector_store = VectorStore(collection_name="demo_collection")
        stats = vector_store.get_collection_stats()
        print(f"Collection created: {stats['collection_name']}")
        print(f"Current documents: {stats['total_documents']}")
        
        # Add demo chunks
        if chunks:
            vector_store.add_documents(chunks)
            updated_stats = vector_store.get_collection_stats()
            print(f"After adding chunks: {updated_stats['total_documents']} documents")
            
            # Demo search
            results = vector_store.search("mechanical ventilation", n_results=2)
            print(f"\nSearch results for 'mechanical ventilation': {len(results)} found")
            if results:
                print(f"Best match: {results[0]['text'][:100]}...")
                print(f"Relevance score: {(1-results[0]['distance']):.2%}")
        
    except Exception as e:
        print(f"Vector store demo failed: {e}")
        print("This is normal if ChromaDB is not properly installed")
    
    # 4. Query Analysis Demo
    print("\n4. 🧠 Query Analysis")
    print("-" * 30)
    
    try:
        # Mock query analysis without loading the full model
        sample_queries = [
            "How do I set PEEP for ARDS patients?",
            "What are the complications of mechanical ventilation?",
            "Emergency intubation procedure steps"
        ]
        
        # Simple pattern matching for demo
        for query in sample_queries:
            print(f"\nQuery: '{query}'")
            
            # Analyze concepts
            concepts = {
                'ventilation': any(term in query.lower() for term in ['ventilation', 'peep', 'intubation']),
                'pathology': any(term in query.lower() for term in ['ards', 'complications']),
                'procedures': any(term in query.lower() for term in ['procedure', 'steps', 'emergency'])
            }
            
            # Analyze question type
            question_type = 'procedural' if any(word in query.lower() for word in ['how', 'steps', 'procedure']) else 'factual'
            
            # Analyze urgency
            urgency = 'high' if 'emergency' in query.lower() else 'medium'
            
            print(f"  Concepts: {[k for k, v in concepts.items() if v]}")
            print(f"  Type: {question_type}")
            print(f"  Urgency: {urgency}")
            
    except Exception as e:
        print(f"Query analysis demo failed: {e}")
    
    # 5. System Requirements
    print("\n5. 💻 System Requirements")
    print("-" * 30)
    
    print("Required for full functionality:")
    print("- Python 3.8+")
    print("- CUDA-capable GPU (recommended)")
    print("- 16GB+ RAM")
    print("- 10GB+ storage for models")
    
    print("\nDependencies:")
    dependencies = [
        "transformers", "torch", "sentence-transformers", 
        "chromadb", "streamlit", "pypdf2"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} (not installed)")
    
    # 6. Usage Examples
    print("\n6. 🚀 Usage Examples")
    print("-" * 30)
    
    print("Command line usage:")
    print("  python main.py                    # Interactive CLI")
    print("  python main.py --gui              # Web interface")
    print("  python main.py --query 'question' # Single query")
    print("  python main.py --help             # Show help")
    
    print("\nWeb interface features:")
    print("- Real-time chat with medical AI")
    print("- Source attribution from textbooks")
    print("- Query analysis and concept detection")
    print("- Conversation history")
    print("- Quick query templates")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("\nNext steps:")
    print("1. Run setup script: setup.bat (Windows) or ./setup.sh (Linux)")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Start the system: python main.py")
    print("\nFor full documentation, see README.md")

if __name__ == "__main__":
    demo_system_overview()
