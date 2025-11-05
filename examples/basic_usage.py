"""
Example usage of the Unified Agent with API-based LLM
Demonstrates how to interact with the agent programmatically
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import UnifiedAgent
from src.rag.llm_engine import LLMEngine
from src.rag.document_processor import VectorStore

# Load environment variables from .env file
load_dotenv()


async def main():
    """Example usage of the unified agent"""
    
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
                # LLM Provider: 'openai', 'anthropic', or 'gemini'
                "llm_provider": os.getenv("LLM_PROVIDER", "google"),
                # Model name (optional, uses defaults if not specified)
                "model_name": os.getenv("LLM_MODEL_NAME", None),
                # API key (optional, reads from environment if not specified)
                "api_key": None,  # Will use OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY from .env
                # Generation parameters
                "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
                "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
                # Vector database configuration
                "vector_db_path": "rag_pipeline/chroma_db",
                "documents_path": "dataset/books"
            }
        },
        "device": "cuda"
    }
    
    print("🚀 Initializing Unified CXR Agent with API-based LLM...")
    print(f"   LLM Provider: {config['models']['rag']['llm_provider']}")
    
    # Initialize components
    vector_store = VectorStore(collection_name="respiratory_care_docs")
    llm_engine = LLMEngine(
        model_name=config["models"]["rag"]["model_name"],
        api_key=config["models"]["rag"]["api_key"],
        max_tokens=config["models"]["rag"]["max_tokens"],
        temperature=config["models"]["rag"]["temperature"]
    )
    
    # Create unified agent
    agent = UnifiedAgent(
        config=config,
        llm_engine=llm_engine,
        vector_store=vector_store
    )
    
    print("✅ Agent initialized!\n")
    
    # Example 1: Ask a medical question (no image)
    print("=" * 80)
    print("Example 1: Medical Question (No Image)")
    print("=" * 80)
    
    response = await agent.process_message(
        query="What are the radiological signs of pneumonia on a chest X-ray?"
    )
    
    print(f"\n📝 Question: What are the radiological signs of pneumonia on a chest X-ray?")
    print(f"\n🤖 Answer:\n{response['answer']}")
    
    if response.get('has_thinking'):
        print(f"\n🧠 AI Thinking:\n{response['thinking'][:200]}...")
    
    if response.get('sources'):
        print(f"\n📚 Sources: {len(response['sources'])} medical references")
    
    print("\n")
    
    # Example 2: Analyze an image with a question
    print("=" * 80)
    print("Example 2: Image Analysis + Question")
    print("=" * 80)
    
    # Note: Replace with actual image path
    image_path = "D:\\3-Clients\\Aerobiosys\\CXR Agent\\examples\\images\\01.jpg"
    
    response = await agent.process_message(
        query="Is this chest X-ray normal or abnormal? What diseases can you detect?",
        image_path=image_path  # Comment this out if you don't have an image
    )
    
    print(f"\n📝 Question: Is this chest X-ray normal or abnormal? What diseases can you detect?")
    print(f"🖼️  Image: {image_path}")
    
    if response.get('image_analysis'):
        print(f"\n🔍 Image Analysis Results:")
        binary = response['image_analysis'].get('binary')
        if binary:
            print(f"   Binary Classification: {binary['prediction']} ({binary['confidence']:.1%})")
            if 'probabilities' in binary:
                print(f"      • Normal: {binary['probabilities']['Normal']:.1%}")
                print(f"      • Abnormal: {binary['probabilities']['Abnormal']:.1%}")
        
        diseases = response['image_analysis'].get('diseases')
        if diseases:
            detected = diseases.get('detected_diseases', {})
            if detected:
                print(f"\n   Detected Diseases ({len(detected)}):")
                for disease, prob in detected.items():
                    print(f"      - {disease}: {prob:.1%}")
            else:
                print(f"\n   Multi-class Classification: No specific diseases detected above threshold")
                # Show top probabilities even if below threshold
                all_probs = diseases.get('all_predictions', {})
                if all_probs:
                    top_3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"      Top 3 probabilities:")
                    for disease, prob in top_3:
                        print(f"         • {disease}: {prob:.1%}")
    
    print(f"\n🤖 Answer:\n{response['answer']}")
    print("\n")
    
    # Example 3: Follow-up question about the image
    print("=" * 80)
    print("Example 3: Follow-up Question (Using Previous Image Context)")
    print("=" * 80)
    
    response = await agent.process_message(
        query="What is the clinical significance of these findings?"
    )
    
    print(f"\n📝 Question: What is the clinical significance of these findings?")
    print(f"\n🤖 Answer:\n{response['answer']}")
    print("\n")
    
    # Example 4: Check conversation history
    print("=" * 80)
    print("Example 4: Conversation History")
    print("=" * 80)
    
    history = agent.get_conversation_history()
    print(f"\n📊 Total conversations: {len(history)}")
    
    for i, conv in enumerate(history, 1):
        print(f"\n--- Conversation {i} ---")
        print(f"Query: {conv['query'][:60]}...")
        print(f"Has Image: {conv['has_image']}")
        print(f"Intent Type: {conv['intent']['type']}")
        print(f"Timestamp: {conv['timestamp']}")
    
    # Example 5: Clear history
    print("\n" + "=" * 80)
    print("Example 5: Clear History")
    print("=" * 80)
    
    agent.clear_history()
    print("✅ Conversation history cleared")
    print(f"📊 Total conversations now: {len(agent.get_conversation_history())}")
    
    print("\n" + "=" * 80)
    print("✅ All examples completed!")
    print("=" * 80)


if __name__ == "__main__":    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
