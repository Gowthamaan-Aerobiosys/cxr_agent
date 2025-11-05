"""
Unified Agent - Central LLM-based orchestrator for all CXR Agent capabilities

This module provides a single conversational interface where users can:
- Upload CXR images and ask questions about them
- Ask general medical questions
- Get automated analysis and explanations

The LLM acts as the central intelligence that routes requests to appropriate models.
"""

import os
import torch
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
from datetime import datetime
import re

from .models.adapters import BinaryClassifierAdapter, MultiClassClassifierAdapter
from .rag.llm_engine import LLMEngine, AgenticRAG
from .rag.document_processor import VectorStore

logger = logging.getLogger(__name__)


class UnifiedAgent:
    """
    Unified conversational agent that uses LLM as central orchestrator
    
    The LLM understands user intent and automatically:
    - Analyzes CXR images when provided
    - Answers medical questions using RAG
    - Combines image analysis with medical knowledge
    - Provides conversational, context-aware responses
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        llm_engine: Optional[LLMEngine] = None,
        vector_store: Optional[VectorStore] = None
    ):
        """
        Initialize the unified agent
        
        Args:
            config: Configuration dictionary
            llm_engine: Pre-initialized LLM engine (optional)
            vector_store: Pre-initialized vector store (optional)
        """
        self.config = config
        self.device = torch.device(
            config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize LLM engine
        if llm_engine:
            self.llm_engine = llm_engine
        else:
            rag_config = config["models"]["rag"]
            
            model_name = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_MODEL")
            
            # Get config values with fallbacks
            max_tokens = rag_config.get("config", {}).get("max_tokens", 2048)
            temperature = rag_config.get("config", {}).get("temperature", 0.7)
            
            self.llm_engine = LLMEngine(
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature
            )
        
        # Initialize vector store for RAG
        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore(collection_name="respiratory_care_docs")
        
        # Initialize vision models (lazy loading)
        self.binary_classifier = None
        self.multiclass_classifier = None
        
        # Conversation history
        self.conversation_history = []
        self.current_image_context = None  # Stores current image analysis
        
        logger.info("Unified Agent initialized successfully")
    
    async def _load_vision_models(self):
        """Lazy load vision models when needed"""
        if self.binary_classifier is None:
            logger.info("Loading binary classifier...")
            binary_config = self.config["models"]["binary_classifier"]
            self.binary_classifier = BinaryClassifierAdapter(
                checkpoint_path=binary_config["checkpoint_path"],
                model_type=binary_config["model_type"],
                device=self.device
            )
            await self.binary_classifier.load()
        
        if self.multiclass_classifier is None:
            logger.info("Loading multi-class classifier...")
            multiclass_config = self.config["models"]["multiclass_classifier"]
            self.multiclass_classifier = MultiClassClassifierAdapter(
                checkpoint_path=multiclass_config["checkpoint_path"],
                num_classes=multiclass_config["num_classes"],
                model_type=multiclass_config["model_type"],
                device=self.device
            )
            await self.multiclass_classifier.load()
    
    def _analyze_user_intent(self, query: str, has_image: bool) -> Dict[str, Any]:
        """
        Analyze what the user wants to do
        
        Returns intent classification:
        - image_analysis: User wants to analyze an image
        - medical_question: User has a general medical question
        - image_question: User is asking about a specific image
        - combined: Needs both image analysis and medical knowledge
        """
        query_lower = query.lower()
        
        intent = {
            "type": "unknown",
            "requires_image_analysis": False,
            "requires_rag": False,
            "specific_model": None,  # 'binary', 'multiclass', or None
            "question_type": "general"
        }
        
        # Keywords for image analysis requests
        image_analysis_keywords = [
            "analyze", "diagnose", "detect", "identify", "classify", 
            "look at", "examine", "check", "scan", "review image",
            "what do you see", "what's in", "show me", "findings"
        ]
        
        # Keywords for binary classification
        binary_keywords = [
            "normal", "abnormal", "healthy", "unhealthy", 
            "is it normal", "is this normal", "normal or abnormal"
        ]
        
        # Keywords for disease detection
        disease_keywords = [
            "disease", "diseases", "pathology", "pathologies", "condition",
            "pneumonia", "effusion", "cardiomegaly", "atelectasis",
            "what disease", "which disease", "any disease"
        ]
        
        # Check if user is asking about an image
        if has_image:
            # User has provided an image
            if any(keyword in query_lower for keyword in image_analysis_keywords):
                intent["requires_image_analysis"] = True
                
                # Check for both binary and disease keywords
                has_binary_keywords = any(keyword in query_lower for keyword in binary_keywords)
                has_disease_keywords = any(keyword in query_lower for keyword in disease_keywords)
                
                if has_binary_keywords and has_disease_keywords:
                    intent["type"] = "image_analysis"
                    intent["specific_model"] = "both"
                    intent["question_type"] = "comprehensive"
                elif has_binary_keywords:
                    intent["type"] = "image_analysis"
                    intent["specific_model"] = "binary"
                    intent["question_type"] = "binary_classification"
                elif has_disease_keywords:
                    intent["type"] = "image_analysis"
                    intent["specific_model"] = "multiclass"
                    intent["question_type"] = "disease_detection"
                else:
                    # General analysis - run both
                    intent["type"] = "image_analysis"
                    intent["specific_model"] = "both"
                    intent["question_type"] = "comprehensive"
            else:
                # Image provided but asking a question about it
                intent["type"] = "combined"
                intent["requires_image_analysis"] = True
                intent["requires_rag"] = True
                intent["specific_model"] = "both"
                intent["question_type"] = "contextual"
        else:
            # No image, check if user is referring to a previous image
            if self.current_image_context and any(
                word in query_lower for word in ["this", "it", "the image", "the scan", "the x-ray", "the cxr"]
            ):
                intent["type"] = "image_question"
                intent["requires_rag"] = True
                intent["question_type"] = "follow_up"
            else:
                # Pure medical question
                intent["type"] = "medical_question"
                intent["requires_rag"] = True
                intent["question_type"] = "knowledge_based"
        
        return intent
    
    async def _analyze_image(self, image_path: str, model_type: str = "both") -> Dict[str, Any]:
        """
        Analyze CXR image using vision models
        
        Args:
            image_path: Path to the image
            model_type: 'binary', 'multiclass', or 'both'
        
        Returns:
            Dictionary with analysis results
        """
        await self._load_vision_models()
        
        results = {
            "image_path": image_path,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if model_type in ["binary", "both"]:
                logger.info("Running binary classification...")
                binary_result = await self.binary_classifier.predict(image_path)
                results["binary_classification"] = binary_result
            
            if model_type in ["multiclass", "both"]:
                logger.info("Running multi-class classification...")
                multiclass_result = await self.multiclass_classifier.predict(
                    image_path, threshold=0.3, top_k=5
                )
                results["multiclass_classification"] = multiclass_result
        
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            results["error"] = str(e)
        
        return results
    
    def _format_image_results_for_llm(self, analysis_results: Dict[str, Any]) -> str:
        """
        Format image analysis results for LLM context
        
        Converts structured analysis into natural text for LLM
        """
        context = "CHEST X-RAY ANALYSIS RESULTS:\n\n"
        
        # Binary classification
        if "binary_classification" in analysis_results:
            binary = analysis_results["binary_classification"]
            prediction = binary.get("prediction", "Unknown")
            confidence = binary.get("confidence", 0)
            context += f"Binary Classification: {prediction} (confidence: {confidence:.1%})\n"
            context += f"  - Normal probability: {binary['probabilities']['Normal']:.1%}\n"
            context += f"  - Abnormal probability: {binary['probabilities']['Abnormal']:.1%}\n\n"
        
        # Multi-class classification
        if "multiclass_classification" in analysis_results:
            multiclass = analysis_results["multiclass_classification"]
            detected = multiclass.get("detected_diseases", {})
            
            if detected:
                context += f"Detected Pathologies ({len(detected)}):\n"
                for disease, prob in sorted(detected.items(), key=lambda x: x[1], reverse=True):
                    context += f"  - {disease}: {prob:.1%}\n"
            else:
                context += "No significant pathologies detected above threshold.\n"
            
            # Add all predictions for context
            all_preds = multiclass.get("all_predictions", {})
            if all_preds:
                context += "\nAll disease probabilities:\n"
                for disease, prob in sorted(all_preds.items(), key=lambda x: x[1], reverse=True)[:5]:
                    context += f"  - {disease}: {prob:.1%}\n"
        
        return context
    
    def _create_unified_prompt(
        self, 
        user_query: str, 
        intent: Dict[str, Any],
        image_context: Optional[str] = None,
        rag_context: Optional[List[Dict[str, Any]]] = None
    ) -> tuple[str, str]:
        """
        Create a unified prompt that combines all contexts
        
        Returns:
            tuple: (system_prompt, user_prompt)
        """
        
        system_prompt = """You are an expert chest X-ray analysis AI assistant with both visual analysis capabilities and deep medical knowledge. You can:

1. Analyze chest X-ray images and detect abnormalities and diseases
2. Provide evidence-based medical information about respiratory conditions
3. Explain radiological findings in clinical context
4. Answer questions about chest X-ray interpretation and respiratory medicine

Guidelines:
- Provide clear, accurate, and clinically relevant information
- Reference the image analysis results when discussing specific findings
- Use medical knowledge to explain the clinical significance
- Always recommend consulting healthcare professionals for patient care decisions
- Be conversational and helpful while maintaining medical accuracy
"""
        
        prompt_parts = []
        
        # Add image context if available
        if image_context:
            prompt_parts.append("=== CHEST X-RAY ANALYSIS ===")
            prompt_parts.append(image_context)
            prompt_parts.append("")
        
        # Add RAG context if available
        if rag_context:
            prompt_parts.append("=== MEDICAL KNOWLEDGE REFERENCES ===")
            for i, doc in enumerate(rag_context, 1):
                source = doc["metadata"].get("source", "Unknown")
                page = doc["metadata"].get("page_number", "Unknown")
                prompt_parts.append(f"Reference {i} (Source: {source}, Page: {page}):")
                prompt_parts.append(doc["text"][:1000])
                prompt_parts.append("")
        
        # Add user query
        prompt_parts.append("=== USER QUESTION ===")
        prompt_parts.append(user_query)
        prompt_parts.append("")
        
        # Add instructions based on intent
        if intent["type"] == "image_analysis":
            prompt_parts.append("Please provide a comprehensive analysis of the chest X-ray findings, explaining what the results mean clinically.")
        elif intent["type"] == "combined":
            prompt_parts.append("Please answer the user's question by combining the image analysis results with relevant medical knowledge.")
        elif intent["type"] == "image_question":
            prompt_parts.append("Please answer the user's question about the previously analyzed image.")
        else:
            prompt_parts.append("Please provide a comprehensive, evidence-based answer to the user's medical question.")
        
        user_prompt = "\n".join(prompt_parts)
        
        return system_prompt, user_prompt
    
    async def process_message(
        self, 
        query: str, 
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point - process any user message with optional image
        
        This is the unified interface that handles everything:
        - Image analysis questions
        - Medical knowledge questions
        - Combined queries
        
        Args:
            query: User's question or request
            image_path: Optional path to CXR image
        
        Returns:
            Unified response with answer, sources, and metadata
        """
        logger.info(f"Processing message: {query[:100]}...")
        if image_path:
            logger.info(f"Image provided: {image_path}")
        
        # Step 1: Analyze intent
        intent = self._analyze_user_intent(query, has_image=bool(image_path))
        logger.info(f"Intent detected: {intent}")
        
        # Step 2: Analyze image if needed
        image_context = None
        image_analysis = None
        
        if intent["requires_image_analysis"] and image_path:
            image_analysis = await self._analyze_image(
                image_path, 
                model_type=intent.get("specific_model", "both")
            )
            image_context = self._format_image_results_for_llm(image_analysis)
            self.current_image_context = image_analysis  # Store for follow-up questions
        elif intent["type"] == "image_question" and self.current_image_context:
            # Use previous image context
            image_context = self._format_image_results_for_llm(self.current_image_context)
            image_analysis = self.current_image_context
        
        # Step 3: Retrieve medical knowledge if needed
        rag_context = None
        if intent["requires_rag"]:
            # Build enhanced query
            search_query = query
            if image_context:
                # Enhance query with image findings
                if image_analysis and "binary_classification" in image_analysis:
                    prediction = image_analysis["binary_classification"].get("prediction")
                    search_query = f"{query} {prediction}"
                if image_analysis and "multiclass_classification" in image_analysis:
                    detected = image_analysis["multiclass_classification"].get("detected_diseases", {})
                    if detected:
                        diseases = " ".join(detected.keys())
                        search_query = f"{query} {diseases}"
            
            logger.info(f"Searching medical knowledge: {search_query}")
            rag_context = self.vector_store.search(search_query, n_results=5)
        
        # Step 4: Create unified prompt
        system_prompt, user_prompt = self._create_unified_prompt(
            user_query=query,
            intent=intent,
            image_context=image_context,
            rag_context=rag_context
        )
        
        # Step 5: Generate LLM response
        logger.info("Generating LLM response...")
        llm_response = self.llm_engine.generate_response(user_prompt, system_prompt)
        
        # Step 6: Format final response
        response = self._format_final_response(
            query=query,
            llm_response=llm_response,
            intent=intent,
            image_analysis=image_analysis,
            rag_context=rag_context
        )
        
        # Step 7: Update conversation history
        self.conversation_history.append({
            "query": query,
            "has_image": bool(image_path),
            "image_path": image_path,
            "intent": intent,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response
    
    def _format_final_response(
        self,
        query: str,
        llm_response: Union[str, Dict[str, Any]],
        intent: Dict[str, Any],
        image_analysis: Optional[Dict[str, Any]],
        rag_context: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Format the final unified response
        
        Returns a structured response that includes:
        - LLM's answer
        - Thinking process (if available)
        - Image analysis results
        - Source references
        - Metadata
        """
        
        # Parse LLM response
        if isinstance(llm_response, dict):
            answer = llm_response.get("final_answer", "")
            thinking = llm_response.get("thinking", "")
            has_thinking = llm_response.get("has_thinking", False)
        else:
            answer = llm_response
            thinking = ""
            has_thinking = False
        
        response = {
            "answer": answer,
            "thinking": thinking,
            "has_thinking": has_thinking,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add image analysis if available
        if image_analysis:
            response["image_analysis"] = {
                "binary": image_analysis.get("binary_classification"),
                "diseases": image_analysis.get("multiclass_classification")
            }
        
        # Add sources if available
        if rag_context:
            response["sources"] = [
                {
                    "source": doc["metadata"]["source"],
                    "page": doc["metadata"]["page_number"],
                    "relevance_score": 1 - doc["distance"]
                }
                for doc in rag_context
            ]
        else:
            response["sources"] = []
        
        return response
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history and image context"""
        self.conversation_history = []
        self.current_image_context = None
        logger.info("Conversation history cleared")
    
    def get_current_image_context(self) -> Optional[Dict[str, Any]]:
        """Get the current image context (if any)"""
        return self.current_image_context
