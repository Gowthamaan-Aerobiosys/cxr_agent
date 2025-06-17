"""
CXR Agent Chat Interface
Comprehensive chat window that combines RAG and CXR analysis capabilities
"""

import streamlit as st
import asyncio
import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np

# Configure page
st.set_page_config(
    page_title="CXR Agent Chat",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import CXR Agent components
try:
    import sys
    sys.path.append(str(Path(__file__).parent))
    from cxr_agent import CXRAgent
    from lung_tools import CXRImageProcessor, CXRClassifier, LungSegmenter
except ImportError as e:
    st.error(f"Failed to import CXR Agent components: {e}")
    st.stop()

class CXRChatInterface:
    """Interactive chat interface for CXR Agent"""
    
    def __init__(self):
        self.initialize_session_state()
        self.initialize_agent()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I'm your CXR Agent assistant. I can help you with:\n\n" +
                              "🔍 **Analyze chest X-rays** - Upload images for classification, segmentation, and pathology detection\n" +
                              "💬 **Answer medical questions** - Ask about respiratory care, ventilation strategies, etc.\n" +
                              "📊 **Generate reports** - Get comprehensive clinical reports\n" +
                              "🧠 **Combine analysis with knowledge** - Get evidence-based interpretations\n\n" +
                              "How can I help you today?"
                }
            ]
        
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        
        if 'uploaded_images' not in st.session_state:
            st.session_state.uploaded_images = {}
        
        if 'chat_mode' not in st.session_state:
            st.session_state.chat_mode = "hybrid"  # hybrid, rag_only, analysis_only
    
    def initialize_agent(self):
        """Initialize CXR Agent"""
        if 'cxr_agent' not in st.session_state:
            try:
                with st.spinner("Initializing CXR Agent..."):
                    st.session_state.cxr_agent = CXRAgent()
                st.success("CXR Agent initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize CXR Agent: {e}")
                st.stop()
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        with st.sidebar:
            st.title("🫁 CXR Agent")
            st.markdown("---")
            
            # Chat mode selection
            st.subheader("Chat Mode")
            st.session_state.chat_mode = st.selectbox(
                "Select mode:",
                options=["hybrid", "rag_only", "analysis_only"],
                format_func=lambda x: {
                    "hybrid": "🔄 Hybrid (RAG + Analysis)",
                    "rag_only": "💬 RAG Only", 
                    "analysis_only": "🔍 Analysis Only"
                }[x],
                index=0
            )
            
            # Image upload section
            st.subheader("📁 Upload CXR Images")
            uploaded_files = st.file_uploader(
                "Choose CXR images",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
                accept_multiple_files=True,
                help="Upload chest X-ray images for analysis"
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.uploaded_images:
                        # Save uploaded file
                        image = Image.open(uploaded_file)
                        st.session_state.uploaded_images[uploaded_file.name] = {
                            'image': image,
                            'uploaded_at': datetime.now(),
                            'analyzed': False
                        }
                        st.success(f"Uploaded: {uploaded_file.name}")
            
            # Show uploaded images
            if st.session_state.uploaded_images:
                st.subheader("📋 Uploaded Images")
                for filename, data in st.session_state.uploaded_images.items():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(filename)
                        st.caption(f"Uploaded: {data['uploaded_at'].strftime('%H:%M:%S')}")
                    with col2:
                        if st.button("🔍", key=f"analyze_{filename}", help="Analyze this image"):
                            self.queue_image_analysis(filename)
            
            st.markdown("---")
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            
            if st.button("🧪 System Status"):
                self.show_system_status()
            
            if st.button("📊 Analysis Summary"):
                self.show_analysis_summary()
            
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = st.session_state.messages[:1]  # Keep welcome message
                st.rerun()
            
            # Sample questions
            st.subheader("💡 Sample Questions")
            sample_questions = [
                "What are the signs of pneumonia on chest X-ray?",
                "How do you differentiate pneumothorax from pneumonia?",
                "What ventilation strategies are used for ARDS?",
                "Explain the pathophysiology of pleural effusion",
                "What are normal lung volumes and capacities?"
            ]
            
            for question in sample_questions:
                if st.button(f"💬 {question[:30]}...", key=f"sample_{hash(question)}"):
                    self.add_message("user", question)
                    asyncio.run(self.process_message(question))
                    st.rerun()
    
    def render_main_chat(self):
        """Render main chat interface"""
        st.title("💬 CXR Agent Chat")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant":
                        self.render_assistant_message(message)
                    else:
                        st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about CXR analysis or medical questions..."):
            self.add_message("user", prompt)
            
            # Process the message
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    response = asyncio.run(self.process_message(prompt))
                    self.add_message("assistant", response)
            
            st.rerun()
    
    def render_assistant_message(self, message: Dict[str, Any]):
        """Render assistant message with rich content"""
        content = message["content"]
        
        # Check if message has special content types
        if isinstance(content, dict):
            # Handle structured responses
            if "text" in content:
                st.markdown(content["text"])
            
            if "analysis_results" in content:
                self.render_analysis_results(content["analysis_results"])
            
            if "images" in content:
                self.render_images(content["images"])
            
            if "charts" in content:
                self.render_charts(content["charts"])
                
        else:
            # Handle plain text
            st.markdown(content)
    
    def render_analysis_results(self, results: Dict[str, Any]):
        """Render CXR analysis results"""
        if not results:
            return
        
        # Classification results
        if "classification" in results:
            st.subheader("🔍 Classification Results")
            classification = results["classification"]
            
            # Create dataframe for better visualization
            df = pd.DataFrame([
                {"Pathology": k.replace("_", " ").title(), "Confidence": v}
                for k, v in classification.items()
                if v > 0.1  # Only show significant results
            ]).sort_values("Confidence", ascending=False)
            
            if not df.empty:
                # Bar chart
                fig = px.bar(
                    df.head(10), 
                    x="Confidence", 
                    y="Pathology",
                    orientation="h",
                    title="Top Classification Results"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Data table
                st.dataframe(df, use_container_width=True)
        
        # Pathology detection
        if "pathology_detection" in results:
            st.subheader("🩺 Pathology Detection")
            pathology = results["pathology_detection"]
            
            if "final_diagnosis" in pathology:
                diagnosis = pathology["final_diagnosis"]
                for condition, assessment in diagnosis.items():
                    if "High probability" in assessment or "Moderate probability" in assessment:
                        if "High probability" in assessment:
                            st.error(f"**{condition.replace('_', ' ').title()}**: {assessment}")
                        else:
                            st.warning(f"**{condition.replace('_', ' ').title()}**: {assessment}")
        
        # Segmentation results
        if "segmentation" in results:
            st.subheader("🫁 Lung Segmentation")
            seg_data = results["segmentation"]
            
            if "features" in seg_data:
                features = seg_data["features"]
                
                # Create metrics for lung features
                col1, col2 = st.columns(2)
                
                for i, (lung, lung_features) in enumerate(features.items()):
                    if lung != "background":
                        with [col1, col2][i % 2]:
                            st.metric(
                                f"{lung.replace('_', ' ').title()} Area",
                                f"{lung_features.get('area', 0):,.0f} pixels"
                            )
                            st.metric(
                                f"{lung.replace('_', ' ').title()} Mean Intensity",
                                f"{lung_features.get('mean_intensity', 0):.1f}"
                            )
        
        # Clinical report
        if "clinical_report" in results:
            st.subheader("📋 Clinical Report")
            report = results["clinical_report"]
            
            for section, content in report.items():
                if section != "disclaimer":
                    with st.expander(f"📄 {section.title()}"):
                        st.text(content)
    
    def render_images(self, images: Dict[str, Any]):
        """Render images with analysis overlays"""
        for image_name, image_data in images.items():
            st.subheader(f"🖼️ {image_name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image_data.get("original"), caption="Original Image")
            
            with col2:
                if "segmentation" in image_data:
                    st.image(image_data["segmentation"], caption="Segmentation Overlay")
                elif "enhanced" in image_data:
                    st.image(image_data["enhanced"], caption="Enhanced Image")
    
    def render_charts(self, charts: Dict[str, Any]):
        """Render additional charts"""
        for chart_name, chart_data in charts.items():
            st.subheader(f"📊 {chart_name}")
            st.plotly_chart(chart_data, use_container_width=True)
    
    def add_message(self, role: str, content: Any):
        """Add message to chat history"""
        st.session_state.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        })
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """Process user message and generate response"""
        try:
            # Determine if this is an image analysis request or RAG query
            is_analysis_request = self.is_image_analysis_request(message)
            
            if st.session_state.chat_mode == "analysis_only" or is_analysis_request:
                return await self.process_analysis_request(message)
            elif st.session_state.chat_mode == "rag_only":
                return await self.process_rag_query(message)
            else:  # hybrid mode
                return await self.process_hybrid_request(message)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "text": f"I apologize, but I encountered an error: {str(e)}. Please try again or contact support."
            }
    
    def is_image_analysis_request(self, message: str) -> bool:
        """Determine if message is requesting image analysis"""
        analysis_keywords = [
            "analyze", "classification", "segment", "pathology", "detect",
            "examine", "review", "diagnosis", "findings", "abnormal",
            "uploaded", "image", "x-ray", "cxr", "chest"
        ]
        
        return any(keyword in message.lower() for keyword in analysis_keywords)
    
    async def process_analysis_request(self, message: str) -> Dict[str, Any]:
        """Process image analysis request"""
        # Check if there are uploaded images to analyze
        if not st.session_state.uploaded_images:
            return {
                "text": "I'd be happy to analyze chest X-rays, but I don't see any uploaded images. Please upload a CXR image using the sidebar, and I'll analyze it for you!"
            }
        
        # Find unanalyzed images or analyze all if specifically requested
        images_to_analyze = []
        if "all" in message.lower() or "uploaded" in message.lower():
            images_to_analyze = list(st.session_state.uploaded_images.keys())
        else:
            # Find the most recent upload
            latest_image = max(
                st.session_state.uploaded_images.items(),
                key=lambda x: x[1]['uploaded_at']
            )
            images_to_analyze = [latest_image[0]]
        
        # Analyze images
        analysis_results = {}
        for filename in images_to_analyze:
            if filename in st.session_state.uploaded_images:
                # Save image temporarily for analysis
                image = st.session_state.uploaded_images[filename]['image']
                temp_path = Path(f"temp_{filename}")
                image.save(temp_path)
                
                try:
                    # Perform analysis
                    results = await st.session_state.cxr_agent.analyze_cxr(
                        image_path=str(temp_path),
                        include_rag=True,
                        generate_report=True
                    )
                    
                    analysis_results[filename] = results
                    st.session_state.uploaded_images[filename]['analyzed'] = True
                    
                    # Store results for later reference
                    st.session_state.analysis_results[filename] = results
                    
                finally:
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()
        
        # Generate response
        if analysis_results:
            # Get the first analysis for detailed response
            first_result = next(iter(analysis_results.values()))
            
            response_text = f"✅ **Analysis Complete for {len(analysis_results)} image(s)**\n\n"
            
            # Add key findings
            if "pathology_detection" in first_result:
                final_diagnosis = first_result["pathology_detection"].get("final_diagnosis", {})
                significant_findings = [
                    condition for condition, assessment in final_diagnosis.items()
                    if "High probability" in assessment or "Moderate probability" in assessment
                ]
                
                if significant_findings:
                    response_text += "🚨 **Key Findings:**\n"
                    for finding in significant_findings[:3]:  # Top 3
                        assessment = final_diagnosis[finding]
                        response_text += f"• {finding.replace('_', ' ').title()}: {assessment}\n"
                else:
                    response_text += "✅ **No significant abnormalities detected**\n"
            
            response_text += "\n📊 **Detailed results are shown below.**"
            
            return {
                "text": response_text,
                "analysis_results": first_result
            }
        else:
            return {
                "text": "❌ I couldn't analyze the images. Please make sure the uploaded files are valid chest X-ray images."
            }
    
    async def process_rag_query(self, message: str) -> Dict[str, Any]:
        """Process RAG-based medical query"""
        try:
            # Check if there are analysis results to provide context
            context = None
            if st.session_state.analysis_results:
                # Use the most recent analysis as context
                latest_analysis = max(
                    st.session_state.analysis_results.items(),
                    key=lambda x: x[1].get('metadata', {}).get('timestamp', '')
                )
                context = latest_analysis[1]
            
            # Query the RAG system
            response = await st.session_state.cxr_agent.query_medical_knowledge(
                question=message,
                context=context
            )
            
            response_text = response.get('response', 'I apologize, but I could not generate a response.')
            
            # Add source information if available
            if 'sources' in response and response['sources']:
                response_text += "\n\n📚 **Sources:**\n"
                for source in response['sources'][:3]:  # Top 3 sources
                    response_text += f"• {source}\n"
            
            return {"text": response_text}
            
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            return {
                "text": f"I encountered an error while processing your question: {str(e)}. Please try rephrasing your question."
            }
    
    async def process_hybrid_request(self, message: str) -> Dict[str, Any]:
        """Process hybrid request (both analysis and RAG)"""
        # Check if this is primarily an analysis request
        if self.is_image_analysis_request(message) and st.session_state.uploaded_images:
            # Do analysis first, then use RAG for interpretation
            analysis_response = await self.process_analysis_request(message)
            
            # If analysis was successful, ask RAG for interpretation
            if "analysis_results" in analysis_response:
                interpretation_query = f"Based on the CXR analysis results, please provide clinical interpretation and recommendations for the following findings: {message}"
                
                rag_response = await self.process_rag_query(interpretation_query)
                
                # Combine responses
                combined_text = analysis_response["text"] + "\n\n" + "🧠 **AI Interpretation:**\n" + rag_response["text"]
                
                return {
                    "text": combined_text,
                    "analysis_results": analysis_response.get("analysis_results")
                }
            else:
                return analysis_response
        else:
            # This is primarily a knowledge query
            return await self.process_rag_query(message)
    
    def queue_image_analysis(self, filename: str):
        """Queue image for analysis via button click"""
        message = f"Please analyze the uploaded image: {filename}"
        self.add_message("user", message)
        
        # Process analysis
        with st.spinner(f"Analyzing {filename}..."):
            response = asyncio.run(self.process_analysis_request(message))
            self.add_message("assistant", response)
        
        st.rerun()
    
    def show_system_status(self):
        """Show system status in sidebar"""
        try:
            status = st.session_state.cxr_agent.get_system_status()
            
            # Add status message to chat
            status_text = "🔧 **System Status:**\n\n"
            
            for component, state in status.get('components', {}).items():
                emoji = "✅" if state == "operational" else "❌"
                status_text += f"{emoji} {component.replace('_', ' ').title()}: {state}\n"
            
            if 'vector_store_stats' in status:
                stats = status['vector_store_stats']
                if 'total_documents' in stats:
                    status_text += f"\n📚 Knowledge Base: {stats['total_documents']} documents"
            
            self.add_message("assistant", {"text": status_text})
            st.rerun()
            
        except Exception as e:
            st.error(f"Error getting system status: {e}")
    
    def show_analysis_summary(self):
        """Show summary of all analyses"""
        if not st.session_state.analysis_results:
            self.add_message("assistant", {"text": "📊 No analyses have been performed yet. Upload and analyze some CXR images first!"})
            st.rerun()
            return
        
        summary_text = f"📊 **Analysis Summary** ({len(st.session_state.analysis_results)} images analyzed)\n\n"
        
        for filename, results in st.session_state.analysis_results.items():
            summary_text += f"🖼️ **{filename}**\n"
            
            # Get key findings
            if "pathology_detection" in results:
                final_diagnosis = results["pathology_detection"].get("final_diagnosis", {})
                significant = [
                    condition for condition, assessment in final_diagnosis.items()
                    if "High probability" in assessment
                ]
                
                if significant:
                    summary_text += f"  🚨 High probability: {', '.join([s.replace('_', ' ') for s in significant])}\n"
                else:
                    summary_text += "  ✅ No high-probability findings\n"
            
            summary_text += "\n"
        
        self.add_message("assistant", {"text": summary_text})
        st.rerun()
    
    def run(self):
        """Run the chat interface"""
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .stChatMessage[data-testid="user"] {
            background-color: #f0f2f6;
        }
        .stChatMessage[data-testid="assistant"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Render interface
        self.render_sidebar()
        self.render_main_chat()

def main():
    """Main function to run the chat interface"""
    chat_interface = CXRChatInterface()
    chat_interface.run()

if __name__ == "__main__":
    main()
