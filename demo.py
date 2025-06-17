#!/usr/bin/env python3
"""
CXR Agent Demo Script
Demonstrates all capabilities of the CXR Agent system including:
- RAG queries for medical knowledge
- CXR image analysis
- Integration between both systems
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from cxr_agent import CXRAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CXRAgentDemo:
    """Demonstration of CXR Agent capabilities"""
    
    def __init__(self):
        self.agent = None
    
    async def initialize_agent(self):
        """Initialize the CXR Agent"""
        print("🚀 Initializing CXR Agent...")
        try:
            self.agent = CXRAgent()
            print("✅ CXR Agent initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize CXR Agent: {e}")
            return False
    
    async def demo_system_status(self):
        """Demonstrate system status check"""
        print("\n" + "="*50)
        print("🔧 SYSTEM STATUS CHECK")
        print("="*50)
        
        try:
            status = self.agent.get_system_status()
            
            print("📊 Component Status:")
            for component, state in status.get('components', {}).items():
                emoji = "✅" if state == "operational" else "❌"
                print(f"  {emoji} {component.replace('_', ' ').title()}: {state}")
            
            if 'vector_store_stats' in status:
                stats = status['vector_store_stats']
                print(f"\n📚 Knowledge Base Statistics:")
                print(f"  • Total documents: {stats.get('total_documents', 'Unknown')}")
                print(f"  • Collection status: {stats.get('collection_status', 'Unknown')}")
            
            print(f"\n⏰ Status checked at: {status.get('timestamp', 'Unknown')}")
            
        except Exception as e:
            print(f"❌ Error checking system status: {e}")
    
    async def demo_rag_queries(self):
        """Demonstrate RAG-based medical queries"""
        print("\n" + "="*50)
        print("💬 RAG MEDICAL KNOWLEDGE QUERIES")
        print("="*50)
        
        # Sample medical questions
        sample_questions = [
            "What are the radiological signs of pneumonia on chest X-ray?",
            "How do you differentiate between pneumothorax and pneumonia?",
            "What are the ventilation strategies for ARDS patients?",
            "Explain the pathophysiology of pleural effusion"
        ]
        
        for i, question in enumerate(sample_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            print("-" * 60)
            
            try:
                response = await self.agent.query_medical_knowledge(
                    question=question,
                    context=None
                )
                
                print(f"🤖 Response: {response.get('response', 'No response generated')}")
                
                # Show sources if available
                if 'sources' in response and response['sources']:
                    print(f"\n📚 Sources ({len(response['sources'])}):")
                    for j, source in enumerate(response['sources'][:3], 1):
                        print(f"  {j}. {source}")
                
                print(f"⏱️  Response time: {response.get('response_time', 'Unknown')} seconds")
                
            except Exception as e:
                print(f"❌ Error processing question: {e}")
            
            # Add spacing between questions
            if i < len(sample_questions):
                input("\nPress Enter to continue to next question...")
    
    async def demo_image_analysis(self):
        """Demonstrate CXR image analysis (mock)"""
        print("\n" + "="*50)
        print("🔍 CXR IMAGE ANALYSIS DEMO")
        print("="*50)
        
        print("📋 This demo shows how CXR analysis would work with real images:")
        print("1. Image preprocessing and enhancement")
        print("2. Multi-pathology classification")
        print("3. Lung segmentation")
        print("4. Feature extraction")
        print("5. Pathology detection")
        print("6. Clinical report generation")
        
        # Create a mock analysis result
        mock_results = {
            "metadata": {
                "image_path": "sample_cxr.jpg",
                "timestamp": datetime.now().isoformat(),
                "processing_time": 2.3
            },
            "preprocessing": {
                "original_size": [1024, 1024],
                "processed_size": [512, 512],
                "enhancement_applied": True
            },
            "classification": {
                "normal": 0.85,
                "pneumonia": 0.12,
                "pleural_effusion": 0.03,
                "pneumothorax": 0.02,
                "cardiomegaly": 0.01
            },
            "segmentation": {
                "lung_area_ratio": 0.68,
                "features": {
                    "left_lung": {
                        "area": 45230,
                        "mean_intensity": 128.5,
                        "std_intensity": 32.1
                    },
                    "right_lung": {
                        "area": 47850,
                        "mean_intensity": 125.2,
                        "std_intensity": 30.8
                    }
                }
            },
            "pathology_detection": {
                "final_diagnosis": {
                    "normal_chest": "High probability (confidence: 0.87) - Normal chest X-ray",
                    "pneumonia": "Low probability (confidence: 0.12) - No significant signs",
                    "pleural_effusion": "Low probability (confidence: 0.03) - No significant signs"
                }
            },
            "clinical_report": {
                "findings": "Chest X-ray shows clear lung fields bilaterally with normal cardiac silhouette.",
                "impression": "Normal chest radiograph. No acute cardiopulmonary findings.",
                "recommendations": "No immediate follow-up required based on imaging findings."
            }
        }
        
        print("\n📊 Mock Analysis Results:")
        print(json.dumps(mock_results, indent=2))
        
        print("\n💡 To analyze real images:")
        print("1. Use the Streamlit chat interface: python run_chat.py")
        print("2. Use the CLI: python cxr_cli.py analyze <image_path>")
        print("3. Use the MCP server API endpoints")
    
    async def demo_hybrid_workflow(self):
        """Demonstrate hybrid analysis + RAG workflow"""
        print("\n" + "="*50)
        print("🧠 HYBRID WORKFLOW DEMO (Analysis + RAG)")
        print("="*50)
        
        print("This demonstrates how analysis results are combined with RAG knowledge:")
        
        # Simulate finding pneumonia in an image
        mock_finding = "pneumonia with consolidation in right lower lobe"
        
        print(f"\n🔍 Simulated finding: {mock_finding}")
        print("\n💬 Asking RAG for clinical interpretation...")
        
        # Query RAG for interpretation
        interpretation_query = (
            f"I found {mock_finding} on a chest X-ray. "
            "Please provide clinical interpretation, differential diagnosis, "
            "and treatment recommendations."
        )
        
        try:
            response = await self.agent.query_medical_knowledge(
                question=interpretation_query,
                context={"finding": mock_finding}
            )
            
            print(f"\n🤖 AI Interpretation:")
            print(response.get('response', 'No response generated'))
            
            if 'sources' in response and response['sources']:
                print(f"\n📚 Evidence sources:")
                for i, source in enumerate(response['sources'][:3], 1):
                    print(f"  {i}. {source}")
        
        except Exception as e:
            print(f"❌ Error getting interpretation: {e}")
    
    async def demo_chat_interface_features(self):
        """Demonstrate chat interface capabilities"""
        print("\n" + "="*50)
        print("💬 CHAT INTERFACE FEATURES")
        print("="*50)
        
        print("The Streamlit chat interface provides:")
        print("\n🔄 **Hybrid Mode** (Default):")
        print("  • Upload CXR images and ask questions about them")
        print("  • Get both technical analysis AND clinical interpretation")
        print("  • Evidence-based recommendations from medical literature")
        
        print("\n💬 **RAG-Only Mode**:")
        print("  • Ask medical questions without image analysis")
        print("  • Get answers from respiratory care textbooks")
        print("  • Perfect for studying or clinical decision support")
        
        print("\n🔍 **Analysis-Only Mode**:")
        print("  • Focus purely on image analysis")
        print("  • Get technical metrics and pathology detection")
        print("  • Ideal for radiology workflow")
        
        print("\n🎯 **Key Features**:")
        print("  • Drag-and-drop image upload")
        print("  • Real-time analysis with progress indicators")
        print("  • Interactive visualizations and charts")
        print("  • Clinical report generation")
        print("  • Chat history and session management")
        print("  • Quick action buttons and sample questions")
        
        print("\n🚀 **To start the chat interface:**")
        print("  python run_chat.py")
        print("  Then open: http://localhost:8501")
    
    async def run_full_demo(self):
        """Run the complete demonstration"""
        print("🫁 CXR AGENT COMPREHENSIVE DEMO")
        print("=" * 60)
        print("This demo showcases the complete CXR Agent system including:")
        print("• RAG-based medical knowledge queries")
        print("• CXR image analysis capabilities")
        print("• Hybrid workflows combining both")
        print("• Interactive chat interface")
        print("=" * 60)
        
        # Initialize agent
        if not await self.initialize_agent():
            return
        
        # Run demonstrations
        await self.demo_system_status()
        
        print("\nPress Enter to continue to RAG demo...")
        input()
        await self.demo_rag_queries()
        
        print("\nPress Enter to continue to image analysis demo...")
        input()
        await self.demo_image_analysis()
        
        print("\nPress Enter to continue to hybrid workflow demo...")
        input()
        await self.demo_hybrid_workflow()
        
        print("\nPress Enter to see chat interface features...")
        input()
        await self.demo_chat_interface_features()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETE!")
        print("🚀 Ready to start the chat interface? Run: python run_chat.py")
        print("=" * 60)

async def main():
    """Main demo function"""
    demo = CXRAgentDemo()
    await demo.run_full_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        logger.exception("Demo failed")
