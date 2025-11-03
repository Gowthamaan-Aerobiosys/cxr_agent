"""
CXR Agent - Unified Conversational Interface
Single chat window where users can ask questions and upload images
The LLM orchestrates everything automatically
"""

import streamlit as st
import os
import sys
import json
import logging
from pathlib import Path
from PIL import Image
from typing import Optional
import asyncio

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import UnifiedAgent
from src.rag.document_processor import DocumentProcessor, VectorStore
from src.rag.llm_engine import LLMEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="CXR Agent - AI Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left-color: #4caf50;
    }
    .thinking-section {
        background-color: #fff8e1;
        border: 1px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        max-height: 300px;
        overflow-y: auto;
    }
    .analysis-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .disease-box {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-card {
        background-color: #f9f9f9;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        border-left: 3px solid #2196f3;
        font-size: 0.9rem;
    }
    .image-preview {
        border: 2px solid #ddd;
        border-radius: 0.5rem;
        padding: 0.5rem;
        margin: 1rem 0;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load application configuration"""
    config_path = "config/mcp_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        return {
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
                    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                    "use_tgi": True,
                    "vector_db_path": "rag_pipeline/chroma_db",
                    "documents_path": "dataset/books"
                }
            },
            "device": "cuda"
        }


@st.cache_resource
def initialize_unified_agent(config):
    """Initialize the unified agent with all capabilities"""
    try:
        st.info("🔄 Initializing AI Assistant... This may take a moment.")
        
        # Initialize document processor and vector store
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        vector_store = VectorStore(collection_name="respiratory_care_docs")
        
        # Check if vector store needs population
        stats = vector_store.get_collection_stats()
        if stats["total_documents"] == 0:
            st.warning("First time setup: Processing medical literature...")
            dataset_path = config["models"]["rag"]["documents_path"]
            if os.path.exists(dataset_path):
                chunks = processor.process_documents(dataset_path)
                if chunks:
                    vector_store.add_documents(chunks, batch_size=50)
                    st.success(f"✅ Processed {len(chunks)} document chunks")
        
        # Initialize LLM engine
        rag_config = config["models"]["rag"]
        
        # Determine provider and model from config or environment
        # Default to OpenAI if API key is available
        provider = os.getenv("LLM_PROVIDER", "google")
        model_name = None  # Will use default for each provider
        
        # Check for API keys and select provider accordingly
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
            model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
            model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        elif os.getenv("GEMINI_API_KEY"):
            provider = "google"
            model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        
        llm_engine = LLMEngine(
            provider=provider,
            model_name=model_name,
            max_tokens=2048,
            temperature=0.7
        )
        
        # Initialize unified agent
        agent = UnifiedAgent(
            config=config,
            llm_engine=llm_engine,
            vector_store=vector_store
        )
        
        st.success("✅ AI Assistant ready!")
        logger.info("Unified agent initialized successfully")
        return agent
        
    except Exception as e:
        st.error(f"❌ Error initializing agent: {str(e)}")
        logger.error(f"Initialization error: {e}", exc_info=True)
        return None


def display_image_analysis(analysis: dict):
    """Display image analysis results in a compact format"""
    
    # Binary classification
    if "binary" in analysis and analysis["binary"]:
        binary = analysis["binary"]
        prediction = binary.get("prediction", "Unknown")
        confidence = binary.get("confidence", 0)
        
        if prediction == "Normal":
            st.markdown(f"""
                <div class="analysis-box">
                    <strong>✓ Classification:</strong> {prediction} ({confidence:.1%} confidence)
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="disease-box">
                    <strong>⚠ Classification:</strong> {prediction} ({confidence:.1%} confidence)
                </div>
            """, unsafe_allow_html=True)
    
    # Disease detection
    if "diseases" in analysis and analysis["diseases"]:
        diseases = analysis["diseases"]
        detected = diseases.get("detected_diseases", {})
        
        if detected:
            disease_list = ", ".join([f"{d} ({p:.1%})" for d, p in detected.items()])
            st.markdown(f"""
                <div class="disease-box">
                    <strong>🔍 Detected Pathologies:</strong><br>
                    {disease_list}
                </div>
            """, unsafe_allow_html=True)


def display_message(message: dict, show_thinking: bool = True):
    """Display a chat message"""
    
    if message["role"] == "user":
        st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
        """, unsafe_allow_html=True)
        
        # Show image if present
        if message.get("image_path"):
            try:
                image = Image.open(message["image_path"])
                st.image(image, caption="Uploaded X-ray", width=300)
            except:
                pass
    
    else:  # assistant
        response = message["content"]
        
        # Show image analysis if present
        if "image_analysis" in response and response["image_analysis"]:
            display_image_analysis(response["image_analysis"])
        
        # Show thinking process
        if show_thinking and response.get("has_thinking") and response.get("thinking"):
            with st.expander("🧠 AI Reasoning Process", expanded=False):
                thinking_text = response['thinking'].replace('\n', '<br>')
                st.markdown(f"""
                    <div class="thinking-section">
                        {thinking_text}
                    </div>
                """, unsafe_allow_html=True)
        
        # Show main answer
        st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 CXR Agent:</strong><br><br>
                {response['answer']}
            </div>
        """, unsafe_allow_html=True)
        
        # Show sources if available
        if response.get("sources") and len(response["sources"]) > 0:
            with st.expander(f"📚 Medical References ({len(response['sources'])})", expanded=False):
                for i, source in enumerate(response["sources"], 1):
                    relevance = source.get("relevance_score", 0)
                    st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {i}:</strong> {source['source']}<br>
                            <strong>Page:</strong> {source['page']}<br>
                            <strong>Relevance:</strong> {relevance:.1%}
                        </div>
                    """, unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 CXR Agent - AI Medical Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Ask questions, upload chest X-rays, get instant AI-powered analysis and insights</p>',
        unsafe_allow_html=True
    )
    
    # Load configuration
    config = load_config()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        llm_options = [
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-2-7b-chat-hf"
        ]
        selected_llm = st.selectbox(
            "Language Model:",
            llm_options,
            index=0,
            help="Choose the AI model for conversation"
        )
        
        if selected_llm != config["models"]["rag"]["model_name"]:
            config["models"]["rag"]["model_name"] = selected_llm
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown(f"**Active Model:** {selected_llm.split('/')[-1]}")
        st.markdown(f"**Device:** {config.get('device', 'cuda')}")
        
        st.markdown("---")
        
        # Display settings
        show_thinking = st.checkbox(
            "Show AI Reasoning",
            value=st.session_state.get("show_thinking", True),
            help="Display the model's thinking process (for DeepSeek)"
        )
        st.session_state.show_thinking = show_thinking
        
        show_analysis_details = st.checkbox(
            "Show Analysis Details",
            value=st.session_state.get("show_analysis", True),
            help="Show detailed image analysis results"
        )
        st.session_state.show_analysis = show_analysis_details
        
        st.markdown("---")
        
        # Quick actions
        st.header("🚀 Quick Actions")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if "agent" in st.session_state and st.session_state.agent:
                st.session_state.agent.clear_history()
            st.rerun()
        
        st.markdown("---")
        
        # Capabilities
        st.header("🎯 Capabilities")
        st.markdown("""
        **I can help you with:**
        - 🖼️ Analyze chest X-rays
        - 🔍 Detect diseases and abnormalities
        - 📚 Answer medical questions
        - 💡 Explain clinical findings
        - 📊 Provide evidence-based insights
        """)
        
        st.markdown("---")
        
        # Example queries
        st.header("💬 Try asking:")
        example_queries = [
            "What are signs of pneumonia?",
            "Explain PEEP settings in ARDS",
            "How to interpret cardiomegaly?",
            "What causes pleural effusion?",
            "Ventilator weaning protocols"
        ]
        for query in example_queries:
            if st.button(query, key=f"example_{query}", use_container_width=True):
                st.session_state.selected_example = query
    
    # Initialize agent
    if "agent" not in st.session_state:
        agent = initialize_unified_agent(config)
        st.session_state.agent = agent
    
    if st.session_state.agent is None:
        st.error("❌ Failed to initialize AI Assistant. Please check configuration and try again.")
        return
    
    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        display_message(message, show_thinking=st.session_state.show_thinking)
    
    # Image upload section
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📤 Upload Chest X-ray (optional)",
            type=['png', 'jpg', 'jpeg', 'dcm'],
            help="Upload a CXR image to analyze it along with your question",
            key="image_upload"
        )
    
    with col2:
        if uploaded_file:
            st.image(uploaded_file, caption="Preview", use_column_width=True)
    
    # Chat input
    st.markdown("---")
    
    # Use example query if selected
    default_value = ""
    if "selected_example" in st.session_state:
        default_value = st.session_state.selected_example
        del st.session_state.selected_example
    
    user_input = st.chat_input(
        "Type your question or request here... (e.g., 'Is this X-ray normal?' or 'What causes pneumothorax?')",
        key="chat_input"
    )
    
    # If example was clicked, use it
    if default_value and not user_input:
        user_input = default_value
    
    # Process input
    if user_input:
        # Save uploaded image temporarily
        temp_image_path = None
        if uploaded_file:
            temp_image_path = f"temp_upload_{uploaded_file.name}"
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "image_path": temp_image_path
        })
        
        # Display user message immediately
        with st.spinner("🤔 Analyzing..."):
            try:
                # Process with unified agent
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    st.session_state.agent.process_message(
                        query=user_input,
                        image_path=temp_image_path
                    )
                )
                loop.close()
                
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Clean up temp file
                if temp_image_path and os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except:
                        pass
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Processing error: {e}", exc_info=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;">
            <strong>⚠️ Medical Disclaimer:</strong> This AI provides educational information only.
            Always consult qualified healthcare professionals for medical decisions.
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
