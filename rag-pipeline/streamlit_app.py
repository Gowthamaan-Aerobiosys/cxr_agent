import streamlit as st
import os
import sys
import json
from pathlib import Path
import logging

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_processor import DocumentProcessor, VectorStore
from qwen_agent import QwenAgent, AgenticRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="CXR Agent - Respiratory Care Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #f0f8ff;
        border-left-color: #4169e1;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left-color: #32cd32;
    }
    .source-card {
        background-color: #fafafa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        border-left: 3px solid #ff6b6b;
    }
    .urgency-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .urgency-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .urgency-low {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the RAG system components"""
    with st.spinner("Initializing Respiratory Care AI System..."):
        try:
            # Initialize components
            processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
            vector_store = VectorStore(collection_name="respiratory_care_docs")
            
            # Check if vector store is empty and needs to be populated
            stats = vector_store.get_collection_stats()
            if stats['total_documents'] == 0:
                st.info("First time setup: Processing medical literature...")
                # Process documents from dataset
                dataset_path = "../dataset/books"
                if os.path.exists(dataset_path):
                    chunks = processor.process_documents(dataset_path)
                    if chunks:
                        vector_store.add_documents(chunks)
                        st.success(f"Successfully processed {len(chunks)} document chunks")
                    else:
                        st.warning("No documents found to process")
                else:
                    st.error(f"Dataset path not found: {dataset_path}")
            else:
                st.info(f"Loaded existing knowledge base with {stats['total_documents']} documents")
            
            # Initialize QWEN agent
            qwen_agent = QwenAgent(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                load_in_4bit=True,
                max_new_tokens=2048
            )
            
            # Initialize agentic RAG
            rag_system = AgenticRAG(vector_store, qwen_agent)
            
            return rag_system, vector_store
            
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            return None, None

def display_source_info(sources):
    """Display source information in a formatted way"""
    st.markdown("**📚 Sources Referenced:**")
    for i, source in enumerate(sources, 1):
        relevance_score = source.get('relevance_score', 0)
        st.markdown(f"""
        <div class="source-card">
            <strong>Source {i}:</strong> {source['source']}<br>
            <strong>Page:</strong> {source['page']}<br>
            <strong>Relevance:</strong> {relevance_score:.2%}
        </div>
        """, unsafe_allow_html=True)

def display_query_analysis(intent_analysis):
    """Display query analysis information"""
    with st.expander("🔍 Query Analysis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Question Type:** {intent_analysis['question_type'].title()}")
            urgency_class = f"urgency-{intent_analysis['urgency']}"
            st.markdown(f"""
            <div class="{urgency_class}" style="padding: 0.5rem; border-radius: 0.3rem; margin: 0.5rem 0;">
                <strong>Urgency Level:</strong> {intent_analysis['urgency'].title()}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Concepts Detected:**")
            concepts = intent_analysis['concepts']
            for concept, detected in concepts.items():
                status = "✅" if detected else "❌"
                st.markdown(f"{status} {concept.title()}")

def main():
    st.markdown('<h1 class="main-header">🫁 CXR Agent - Respiratory Care Assistant</h1>', 
                unsafe_allow_html=True)
    
    # Initialize system
    if 'rag_system' not in st.session_state:
        rag_system, vector_store = initialize_system()
        st.session_state.rag_system = rag_system
        st.session_state.vector_store = vector_store
    
    if st.session_state.rag_system is None:
        st.error("Failed to initialize the system. Please check the logs and try again.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Information")
        
        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_collection_stats()
            st.metric("Knowledge Base Size", f"{stats['total_documents']} documents")
        
        st.header("📋 Quick Actions")
        
        # Predefined queries for quick access
        quick_queries = [
            "What are the indications for mechanical ventilation?",
            "How do you set PEEP in ARDS patients?",
            "Explain ventilator weaning protocols",
            "What are the complications of mechanical ventilation?",
            "How to troubleshoot high peak pressures?",
            "BiPAP vs CPAP differences and indications",
            "ABG interpretation in respiratory failure",
            "Ventilator modes: Volume vs Pressure control"
        ]
        
        selected_query = st.selectbox(
            "Select a common query:",
            [""] + quick_queries
        )
        
        if st.button("Use Selected Query") and selected_query:
            st.session_state.user_input = selected_query
        
        st.header("🗂️ Conversation History")
        if hasattr(st.session_state.rag_system, 'conversation_history'):
            history_count = len(st.session_state.rag_system.conversation_history)
            st.metric("Total Queries", history_count)
            
            if st.button("Clear History"):
                st.session_state.rag_system.clear_history()
                st.success("History cleared!")
                st.rerun()
    
    # Main interface
    st.markdown("### 💬 Ask your respiratory care question:")
    
    # Initialize session state for chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # User input
    user_input = st.text_input(
        "Your question:",
        key="user_input",
        placeholder="e.g., What are the best ventilator settings for a COPD patient?",
        help="Ask any question about respiratory care, mechanical ventilation, or pulmonary medicine"
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary")
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Process query
    if ask_button and user_input:
        with st.spinner("Analyzing your question and searching medical literature..."):
            try:
                # Add user message to chat
                st.session_state.messages.append({
                    "role": "user", 
                    "content": user_input
                })
                
                # Process query
                response = st.session_state.rag_system.process_query(user_input)
                
                # Add assistant response to chat
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
                # Clear input
                st.session_state.user_input = ""
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    # Display chat history
    if st.session_state.messages:
        st.markdown("### 💭 Conversation")
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👨‍⚕️ You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            
            else:  # assistant message
                response_data = message["content"]
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 CXR Agent:</strong><br>
                    {response_data['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Display additional information
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    display_source_info(response_data['sources'])
                
                with col2:
                    display_query_analysis({
                        'question_type': response_data['query_type'],
                        'urgency': response_data['urgency'],
                        'concepts': response_data['concepts_detected']
                    })
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <strong>⚠️ Medical Disclaimer:</strong> This AI assistant provides educational information only. 
        Always consult with qualified healthcare professionals for patient-specific medical decisions.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
