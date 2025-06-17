import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    GenerationConfig
)
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class QwenAgent:
    """QWEN 2.5 model wrapper for agentic RAG"""
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 load_in_4bit: bool = True,
                 max_new_tokens: int = 2048):
        
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        # Configure quantization for memory efficiency
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
        
        logger.info(f"Loading QWEN model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Generation config
        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        logger.info("QWEN model loaded successfully")
    
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
            source = doc['metadata'].get('source', 'Unknown')
            page = doc['metadata'].get('page_number', 'Unknown')
            context_text += f"\n--- Reference {i} (Source: {source}, Page: {page}) ---\n"
            context_text += doc['text'][:1500]  # Limit context length
            context_text += "\n"
        
        prompt = f"""<|im_start|>system
{self.format_system_prompt()}<|im_end|>
<|im_start|>user
Based on the following medical literature context, please answer the question thoroughly and accurately.

CONTEXT:
{context_text}

QUESTION: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't fully address the question, indicate what additional information might be needed.<|im_end|>
<|im_start|>assistant"""
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using QWEN model"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user query to determine intent and extract key concepts"""
        
        # Medical concept patterns
        ventilation_patterns = [
            r'\b(mechanical ventilation|ventilat|PEEP|pressure support|volume control)\b',
            r'\b(BiPAP|CPAP|NIV|non-invasive)\b',
            r'\b(weaning|extubat|liberation)\b'
        ]
        
        pathology_patterns = [
            r'\b(COPD|asthma|pneumonia|ARDS|respiratory failure)\b',
            r'\b(pneumothorax|pleural effusion|pulmonary edema)\b'
        ]
        
        procedure_patterns = [
            r'\b(intubation|tracheostomy|bronchoscopy)\b',
            r'\b(arterial blood gas|ABG|spirometry)\b'
        ]
        
        # Analyze intent
        intent_analysis = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'concepts': {
                'ventilation': any(re.search(pattern, query, re.IGNORECASE) for pattern in ventilation_patterns),
                'pathology': any(re.search(pattern, query, re.IGNORECASE) for pattern in pathology_patterns),
                'procedures': any(re.search(pattern, query, re.IGNORECASE) for pattern in procedure_patterns)
            },
            'question_type': self._classify_question_type(query),
            'urgency': self._assess_urgency(query)
        }
        
        return intent_analysis
    
    def _classify_question_type(self, query: str) -> str:
        """Classify the type of question being asked"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'procedure', 'steps', 'process']):
            return 'procedural'
        elif any(word in query_lower for word in ['why', 'mechanism', 'physiology', 'pathophysiology']):
            return 'explanatory'
        elif any(word in query_lower for word in ['when', 'indication', 'contraindication']):
            return 'clinical_decision'
        elif any(word in query_lower for word in ['what', 'definition', 'normal values']):
            return 'factual'
        elif any(word in query_lower for word in ['troubleshoot', 'problem', 'alarm', 'issue']):
            return 'troubleshooting'
        else:
            return 'general'
    
    def _assess_urgency(self, query: str) -> str:
        """Assess urgency level of the query"""
        urgent_keywords = ['emergency', 'urgent', 'critical', 'alarm', 'crisis', 'immediately']
        routine_keywords = ['routine', 'general', 'education', 'learning']
        
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in urgent_keywords):
            return 'high'
        elif any(keyword in query_lower for keyword in routine_keywords):
            return 'low'
        else:
            return 'medium'

class AgenticRAG:
    """Main agentic RAG pipeline orchestrator"""
    
    def __init__(self, vector_store, qwen_agent: QwenAgent):
        self.vector_store = vector_store
        self.qwen_agent = qwen_agent
        self.conversation_history = []
    
    def process_query(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Process user query through the agentic RAG pipeline"""
        
        # Step 1: Analyze query intent
        intent_analysis = self.qwen_agent.analyze_query_intent(query)
        logger.info(f"Query intent: {intent_analysis['question_type']}")
        
        # Step 2: Retrieve relevant documents
        relevant_docs = self.vector_store.search(query, n_results=n_results)
        
        # Step 3: Generate enhanced query if needed
        enhanced_query = self._enhance_query(query, intent_analysis)
        
        # Step 4: Re-retrieve with enhanced query if different
        if enhanced_query != query:
            enhanced_docs = self.vector_store.search(enhanced_query, n_results=n_results)
            # Combine and deduplicate results
            all_docs = relevant_docs + enhanced_docs
            seen_ids = set()
            unique_docs = []
            for doc in all_docs:
                if doc['id'] not in seen_ids:
                    unique_docs.append(doc)
                    seen_ids.add(doc['id'])
            relevant_docs = unique_docs[:n_results]
        
        # Step 5: Generate response using QWEN
        prompt = self.qwen_agent.format_rag_prompt(query, relevant_docs)
        response = self.qwen_agent.generate_response(prompt)
        
        # Step 6: Post-process and format response
        formatted_response = self._format_response(response, relevant_docs, intent_analysis)
        
        # Step 7: Update conversation history
        self.conversation_history.append({
            'query': query,
            'response': formatted_response,
            'timestamp': datetime.now().isoformat(),
            'intent': intent_analysis,
            'sources_used': len(relevant_docs)
        })
        
        return formatted_response
    
    def _enhance_query(self, original_query: str, intent_analysis: Dict[str, Any]) -> str:
        """Enhance query based on intent analysis"""
        
        # Add domain-specific terms based on detected concepts
        enhancements = []
        
        if intent_analysis['concepts']['ventilation']:
            enhancements.extend(['mechanical ventilation', 'respiratory mechanics', 'ventilator settings'])
        
        if intent_analysis['concepts']['pathology']:
            enhancements.extend(['pathophysiology', 'clinical presentation', 'diagnosis'])
        
        if intent_analysis['concepts']['procedures']:
            enhancements.extend(['clinical procedure', 'technique', 'indications'])
        
        if enhancements:
            enhanced_query = f"{original_query} {' '.join(enhancements)}"
            return enhanced_query
        
        return original_query
    
    def _format_response(self, response: str, sources: List[Dict[str, Any]], 
                        intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Format the final response with metadata"""
        
        return {
            'answer': response,
            'query_type': intent_analysis['question_type'],
            'urgency': intent_analysis['urgency'],
            'sources': [
                {
                    'source': doc['metadata']['source'],
                    'page': doc['metadata']['page_number'],
                    'relevance_score': 1 - doc['distance']  # Convert distance to similarity
                } for doc in sources
            ],
            'concepts_detected': intent_analysis['concepts'],
            'timestamp': datetime.now().isoformat()
        }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
