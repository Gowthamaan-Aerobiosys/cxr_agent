import os
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import re
from enum import Enum

# API clients for different LLM providers
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.genai as genai
except ImportError:
    genai = None

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class LLMEngine:
    """API-based wrapper for external LLM providers (OpenAI, Anthropic, Google)"""

    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        """
        Initialize LLM Engine with API-based provider
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'gemini')
            model_name: Specific model name (defaults to recommended for each provider)
            api_key: API key (if not set via environment variables)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)
        """
        self.provider = LLMProvider(provider.lower())
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Set default model names if not provided
        if model_name is None:
            model_name = self._get_default_model()
        self.model_name = model_name
        
        # Initialize the appropriate client
        self._init_client(api_key)
        
        logger.info(f"Initialized {self.provider.value} LLM Engine with model: {self.model_name}")

    def _get_default_model(self) -> str:
        """Get default model for each provider"""
        defaults = {
            LLMProvider.OPENAI: "gpt-4-turbo-preview",
            LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
            LLMProvider.GOOGLE: "gemini-2.5-pro",
        }
        return defaults[self.provider]

    def _init_client(self, api_key: Optional[str] = None):
        """Initialize API client for the selected provider"""
        if self.provider == LLMProvider.OPENAI:
            if openai is None:
                raise ImportError("openai package not installed. Run: pip install openai")
            
            # Set API key from parameter or environment
            if api_key:
                openai.api_key = api_key
            else:
                openai.api_key = os.getenv("OPENAI_API_KEY")
            
            if not openai.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter")
            
            self.client = openai.OpenAI(api_key=openai.api_key)
            
        elif self.provider == LLMProvider.ANTHROPIC:
            if anthropic is None:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            
            # Get API key from parameter or environment
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter")
            
            self.client = anthropic.Anthropic(api_key=key)
            
        elif self.provider == LLMProvider.GOOGLE:
            if genai is None:
                raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
            
            # Get API key - try both GEMINI_API_KEY and GOOGLE_API_KEY for backwards compatibility
            key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError("Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable or pass api_key parameter")
            
            # Configure Gemini with API key
            genai.configure(api_key=key)
            self.client = genai.GenerativeModel(self.model_name)

    def format_system_prompt(self) -> str:
        """Create system prompt for respiratory care agent"""
        return """You are an expert respiratory care AI assistant with deep knowledge of mechanical ventilation, pulmonary medicine, and respiratory therapy. Your role is to:

1. Provide accurate, evidence-based answers about respiratory care
2. Help clinicians understand ventilation strategies and patient management
3. Explain complex physiological concepts clearly
4. Reference relevant clinical guidelines and best practices
5. Always prioritize patient safety in your recommendations

Guidelines:
- Base your answers on the provided medical literature context
- If you're uncertain, clearly state limitations
- Always recommend consulting with physicians for patient-specific decisions
- Use appropriate medical terminology while remaining accessible
- Provide practical, actionable guidance when appropriate

You have access to comprehensive respiratory care textbooks and clinical references."""

    def format_rag_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Format prompt with RAG context"""
        
        # Format context documents
        context_text = ""
        for i, doc in enumerate(context_docs, 1):
            source = doc["metadata"].get("source", "Unknown")
            page = doc["metadata"].get("page_number", "Unknown")
            context_text += (
                f"\n--- Reference {i} (Source: {source}, Page: {page}) ---\n"
            )
            context_text += doc["text"][:1500]  # Limit context length
            context_text += "\n"

        user_prompt = f"""Based on the following medical literature context, please answer the question thoroughly and accurately.

CONTEXT:
{context_text}

QUESTION: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't fully address the question, indicate what additional information might be needed."""

        return user_prompt

    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user query to determine intent and extract key concepts"""

        # Medical concept patterns
        ventilation_patterns = [
            r"\b(mechanical ventilation|ventilat|PEEP|pressure support|volume control)\b",
            r"\b(BiPAP|CPAP|NIV|non-invasive)\b",
            r"\b(weaning|extubat|liberation)\b",
        ]

        pathology_patterns = [
            r"\b(COPD|asthma|pneumonia|ARDS|respiratory failure)\b",
            r"\b(pneumothorax|pleural effusion|pulmonary edema)\b",
        ]

        procedure_patterns = [
            r"\b(intubation|tracheostomy|bronchoscopy)\b",
            r"\b(arterial blood gas|ABG|spirometry)\b",
        ]

        # Analyze intent
        intent_analysis = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "concepts": {
                "ventilation": any(
                    re.search(pattern, query, re.IGNORECASE)
                    for pattern in ventilation_patterns
                ),
                "pathology": any(
                    re.search(pattern, query, re.IGNORECASE)
                    for pattern in pathology_patterns
                ),
                "procedures": any(
                    re.search(pattern, query, re.IGNORECASE)
                    for pattern in procedure_patterns
                ),
            },
            "question_type": self._classify_question_type(query),
            "urgency": self._assess_urgency(query),
        }

        return intent_analysis

    def _classify_question_type(self, query: str) -> str:
        """Classify the type of question being asked"""
        query_lower = query.lower()

        if any(
            word in query_lower for word in ["how", "procedure", "steps", "process"]
        ):
            return "procedural"
        elif any(
            word in query_lower
            for word in ["why", "mechanism", "physiology", "pathophysiology"]
        ):
            return "explanatory"
        elif any(
            word in query_lower for word in ["when", "indication", "contraindication"]
        ):
            return "clinical_decision"
        elif any(
            word in query_lower for word in ["what", "definition", "normal values"]
        ):
            return "factual"
        elif any(
            word in query_lower
            for word in ["troubleshoot", "problem", "alarm", "issue"]
        ):
            return "troubleshooting"
        else:
            return "general"

    def _assess_urgency(self, query: str) -> str:
        """Assess urgency level of the query"""
        urgent_keywords = [
            "emergency",
            "urgent",
            "critical",
            "alarm",
            "crisis",
            "immediately",
        ]
        routine_keywords = ["routine", "general", "education", "learning"]

        query_lower = query.lower()

        if any(keyword in query_lower for keyword in urgent_keywords):
            return "high"
        elif any(keyword in query_lower for keyword in routine_keywords):
            return "low"
        else:
            return "medium"

    def generate_response(self, user_prompt: str) -> Dict[str, str]:
        """Generate response using the selected LLM provider API"""
        try:
            system_prompt = self.format_system_prompt()
            
            if self.provider == LLMProvider.OPENAI:
                response = self._generate_openai(system_prompt, user_prompt)
            elif self.provider == LLMProvider.ANTHROPIC:
                response = self._generate_anthropic(system_prompt, user_prompt)
            elif self.provider == LLMProvider.GOOGLE:
                response = self._generate_gemini(system_prompt, user_prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            return {
                "final_answer": response.strip(),
                "thinking": "",
                "has_thinking": False
            }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_msg = f"I apologize, but I encountered an error while processing your request: {str(e)}"
            return {
                "final_answer": error_msg,
                "thinking": "",
                "has_thinking": False
            }

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    def _generate_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Anthropic Claude API"""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise

    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> str:
        """Generate response using Google Gemini API"""
        try:
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Configure generation settings
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
            
            # Generate response
            response = self.client.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise


class AgenticRAG:
    """Main agentic RAG pipeline orchestrator"""

    def __init__(self, vector_store, llm_agent: LLMEngine):
        self.vector_store = vector_store
        self.llm_agent = llm_agent
        self.conversation_history = []

    def process_query(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Process user query through the agentic RAG pipeline"""

        # Step 1: Analyze query intent
        intent_analysis = self.llm_agent.analyze_query_intent(query)
        logger.info(f"Query intent: {intent_analysis['question_type']}")

        # Step 2: Retrieve relevant documents
        relevant_docs = self.vector_store.search(query, n_results=n_results)

        # Step 3: Generate enhanced query if needed
        enhanced_query = self._enhance_query(query, intent_analysis)

        # Step 4: Re-retrieve with enhanced query if different
        if enhanced_query != query:
            enhanced_docs = self.vector_store.search(
                enhanced_query, n_results=n_results
            )
            # Combine and deduplicate results
            all_docs = relevant_docs + enhanced_docs
            seen_ids = set()
            unique_docs = []
            for doc in all_docs:
                if doc["id"] not in seen_ids:
                    unique_docs.append(doc)
                    seen_ids.add(doc["id"])
            relevant_docs = unique_docs[:n_results]
        
        logger.info("Generating LLM response")
        # Step 5: Generate response using LLM API
        prompt = self.llm_agent.format_rag_prompt(query, relevant_docs)
        response = self.llm_agent.generate_response(prompt)

        # Step 6: Post-process and format response
        formatted_response = self._format_response(
            response, relevant_docs, intent_analysis
        )

        # Step 7: Update conversation history
        self.conversation_history.append(
            {
                "query": query,
                "response": formatted_response,
                "timestamp": datetime.now().isoformat(),
                "intent": intent_analysis,
                "sources_used": len(relevant_docs),
            }
        )

        return formatted_response

    def _enhance_query(
        self, original_query: str, intent_analysis: Dict[str, Any]
    ) -> str:
        """Enhance query based on intent analysis"""

        # Add domain-specific terms based on detected concepts
        enhancements = []

        if intent_analysis["concepts"]["ventilation"]:
            enhancements.extend(
                [
                    "mechanical ventilation",
                    "respiratory mechanics",
                    "ventilator settings",
                ]
            )

        if intent_analysis["concepts"]["pathology"]:
            enhancements.extend(
                ["pathophysiology", "clinical presentation", "diagnosis"]
            )

        if intent_analysis["concepts"]["procedures"]:
            enhancements.extend(["clinical procedure", "technique", "indications"])

        if enhancements:
            enhanced_query = f"{original_query} {' '.join(enhancements)}"
            return enhanced_query

        return original_query

    def _format_response(
        self,
        response: Dict[str, str],
        sources: List[Dict[str, Any]],
        intent_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Format the final response with metadata"""
        
        # Handle both dictionary and string responses for backward compatibility
        if isinstance(response, dict):
            answer = response.get("final_answer", "")
            thinking = response.get("thinking", "")
            has_thinking = response.get("has_thinking", False)
        else:
            answer = response
            thinking = ""
            has_thinking = False

        return {
            "answer": answer,
            "thinking": thinking,
            "has_thinking": has_thinking,
            "query_type": intent_analysis["question_type"],
            "urgency": intent_analysis["urgency"],
            "sources": [
                {
                    "source": doc["metadata"]["source"],
                    "page": doc["metadata"]["page_number"],
                    "relevance_score": 1
                    - doc["distance"],  # Convert distance to similarity
                }
                for doc in sources
            ],
            "concepts_detected": intent_analysis["concepts"],
            "timestamp": datetime.now().isoformat(),
        }

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

