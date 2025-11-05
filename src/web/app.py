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

# Custom CSS for modern UI
st.markdown("""
<style>
    /* General body styling */
    body {
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
        background-color: #f0f2f6;
    }

    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a73e8;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #5f6368;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Styling for Streamlit's chat messages */
    [data-testid="stChatMessage"] {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    /* User message styling */
    [data-testid="stChatMessage"]:has([data-testid="stMarkdownContainer"] p:first-child:contains("You:")) {
        background-color: #e3f2fd;
        border-left: 5px solid #1a73e8;
    }

    /* Assistant message styling */
    [data-testid="stChatMessage"]:has([data-testid="stMarkdownContainer"] p:first-child:contains("CXR Agent:")) {
        background-color: #ffffff;
        border-left: 5px solid #34a853;
    }

    /* Thinking process section */
    .thinking-section {
        background-color: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        max-height: 300px;
        overflow-y: auto;
        color: #1f2937;
    }

    /* Analysis and disease boxes */
    .analysis-box, .disease-box {
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        border-left-width: 5px;
        border-left-style: solid;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .analysis-box {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
        color: #1b5e20;
    }
    .disease-box {
        background-color: #ffebee;
        border-left-color: #f44336;
        color: #c62828;
    }

    /* Source card styling */
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
        font-size: 0.95rem;
        color: #343a40;
        transition: all 0.2s ease-in-out;
    }
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Image preview in chat */
    .image-preview {
        border: 2px solid #e0e0e0;
        border-radius: 0.75rem;
        padding: 0.5rem;
        margin-top: 1rem;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
                    "vector_db_path": "chroma_db",
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
        
        # The LLM Engine is now specific to Google Gemini
        model_name = config["models"]["rag"].get("model_name") or os.getenv("GEMINI_MODEL")

        llm_engine = LLMEngine(
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
    """Display a chat message using Streamlit's native chat elements."""
    
    role = message["role"]
    avatar = "👤" if role == "user" else "🤖"
    
    with st.chat_message(role, avatar=avatar):
        # Display content
        st.markdown(f"**{'You' if role == 'user' else 'CXR Agent'}:**")
        
        if role == "user":
            st.markdown(message["content"])
            if message.get("image_path"):
                try:
                    image = Image.open(message["image_path"])
                    st.image(image, caption="Uploaded X-ray", width=200, clamp=True)
                except Exception as e:
                    logger.warning(f"Could not display user image: {e}")
        
        else:  # Assistant
            response = message["content"]
            
            # Show image analysis if present
            if "image_analysis" in response and response["image_analysis"]:
                display_image_analysis(response["image_analysis"])
            
            # Show thinking process
            if show_thinking and response.get("has_thinking") and response.get("thinking"):
                with st.expander("🧠 AI Reasoning Process", expanded=False):
                    thinking_text = response['thinking'].replace('\n', '<br>')
                    st.markdown(f'<div class="thinking-section">{thinking_text}</div>', unsafe_allow_html=True)
            
            # Show main answer
            st.markdown(response.get('answer', ''))
            
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
            "gemini-2.5-pro"
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
        display_message(message)

    # --- Chat Input and File Uploader at the bottom ---
    st.markdown("---")

    # We use a form to handle the file upload and text input together
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "Type your question or request here...",
                value="",
                placeholder="Ask about an X-ray or a medical topic...",
                label_visibility="collapsed"
            )
        with col2:
            uploaded_file = st.file_uploader(
                "Upload CXR",
                type=['png', 'jpg', 'jpeg', 'dcm'],
                label_visibility="collapsed",
                key="file_uploader_form"
            )
        
        submit_button = st.form_submit_button(label='Send')

    # Process input when form is submitted
    if submit_button and (user_input or uploaded_file):
        # Save uploaded image temporarily
        temp_image_path = None
        if uploaded_file:
            # Ensure temp directory exists
            if not os.path.exists("temp"):
                os.makedirs("temp")
            temp_image_path = os.path.join("temp", f"upload_{uploaded_file.name}")
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input or "Analyzing uploaded image...",
            "image_path": temp_image_path
        })
        
        # Rerun to display the user's message immediately
        st.rerun()

    # Check if the last message was from the user and needs processing
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and "content" in st.session_state.messages[-1]:
        last_message = st.session_state.messages[-1]
        
        # A simple flag to prevent re-processing
        if not last_message.get("processed", False):
            query = last_message["content"]
            image_path = last_message.get("image_path")

            with st.spinner("🤔 Analyzing..."):
                try:
                    # Process with unified agent
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(
                        st.session_state.agent.process_message(
                            query=query,
                            image_path=image_path
                        )
                    )
                    loop.close()
                    
                    # Add assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    # Mark the user message as processed
                    st.session_state.messages[-2]["processed"] = True
                    
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
