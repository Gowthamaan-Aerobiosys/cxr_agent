import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Configuration for API-based LLM"""
    provider: str = "openai"  # 'openai', 'anthropic', or 'gemini'
    model_name: Optional[str] = None  # If None, uses provider's default
    api_key: Optional[str] = None  # If None, reads from environment
    max_tokens: int = 2048
    temperature: float = 0.7

@dataclass
class DocumentConfig:
    """Configuration for document processing"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    supported_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.pdf']

@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    collection_name: str = "respiratory_care_docs"
    embedding_model: str = "all-MiniLM-L6-v2"
    persist_directory: str = "./chroma_db"
    n_results_default: int = 5
    max_results: int = 10

@dataclass
class AgentConfig:
    """Configuration for agentic behavior"""
    max_context_length: int = 4096
    response_max_length: int = 1500
    enable_query_enhancement: bool = True
    enable_intent_analysis: bool = True
    conversation_memory: bool = True

@dataclass
class SystemConfig:
    """Main system configuration"""
    dataset_path: str = "../dataset/books"
    log_level: str = "INFO"
    debug_mode: bool = False
    
    # Component configs
    model: ModelConfig = None
    document: DocumentConfig = None
    vector_store: VectorStoreConfig = None
    agent: AgentConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.document is None:
            self.document = DocumentConfig()
        if self.vector_store is None:
            self.vector_store = VectorStoreConfig()
        if self.agent is None:
            self.agent = AgentConfig()

# Medical domain-specific configurations
MEDICAL_CONCEPTS = {
    'ventilation': [
        'mechanical ventilation', 'ventilator', 'PEEP', 'pressure support',
        'volume control', 'BiPAP', 'CPAP', 'NIV', 'non-invasive',
        'weaning', 'extubation', 'liberation'
    ],
    'pathology': [
        'COPD', 'asthma', 'pneumonia', 'ARDS', 'respiratory failure',
        'pneumothorax', 'pleural effusion', 'pulmonary edema'
    ],
    'procedures': [
        'intubation', 'tracheostomy', 'bronchoscopy',
        'arterial blood gas', 'ABG', 'spirometry'
    ],
    'physiology': [
        'gas exchange', 'ventilation perfusion', 'compliance',
        'resistance', 'dead space', 'shunt'
    ]
}

URGENCY_KEYWORDS = {
    'high': ['emergency', 'urgent', 'critical', 'alarm', 'crisis', 'immediately'],
    'low': ['routine', 'general', 'education', 'learning', 'background']
}

QUESTION_TYPE_PATTERNS = {
    'procedural': ['how', 'procedure', 'steps', 'process', 'method'],
    'explanatory': ['why', 'mechanism', 'physiology', 'pathophysiology', 'explain'],
    'clinical_decision': ['when', 'indication', 'contraindication', 'should'],
    'factual': ['what', 'definition', 'normal values', 'range'],
    'troubleshooting': ['troubleshoot', 'problem', 'alarm', 'issue', 'fix']
}

# Streamlit app configuration
STREAMLIT_CONFIG = {
    'page_title': "CXR Agent - Respiratory Care Assistant",
    'page_icon': "🫁",
    'layout': "wide",
    'initial_sidebar_state': "expanded"
}

QUICK_QUERIES = [
    "What are the indications for mechanical ventilation?",
    "How do you set PEEP in ARDS patients?",
    "Explain ventilator weaning protocols",
    "What are the complications of mechanical ventilation?",
    "How to troubleshoot high peak pressures?",
    "BiPAP vs CPAP differences and indications",
    "ABG interpretation in respiratory failure",
    "Ventilator modes: Volume vs Pressure control",
    "What are the signs of ventilator-associated pneumonia?",
    "How to manage patient-ventilator asynchrony?"
]

# Default system configuration instance
DEFAULT_CONFIG = SystemConfig()

def load_config_from_env() -> SystemConfig:
    """Load configuration from environment variables"""
    config = SystemConfig()
    
    # Model configuration
    config.model.provider = os.getenv('LLM_PROVIDER', config.model.provider)
    config.model.model_name = os.getenv('LLM_MODEL_NAME', config.model.model_name)
    config.model.api_key = os.getenv('LLM_API_KEY', config.model.api_key)
    config.model.max_tokens = int(os.getenv('LLM_MAX_TOKENS', str(config.model.max_tokens)))
    config.model.temperature = float(os.getenv('LLM_TEMPERATURE', str(config.model.temperature)))
    
    # Document configuration
    config.document.chunk_size = int(os.getenv('CHUNK_SIZE', str(config.document.chunk_size)))
    config.document.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', str(config.document.chunk_overlap)))
    
    # Vector store configuration
    config.vector_store.collection_name = os.getenv('COLLECTION_NAME', config.vector_store.collection_name)
    config.vector_store.embedding_model = os.getenv('EMBEDDING_MODEL', config.vector_store.embedding_model)
    
    # System configuration
    config.dataset_path = os.getenv('DATASET_PATH', config.dataset_path)
    config.log_level = os.getenv('LOG_LEVEL', config.log_level)
    config.debug_mode = os.getenv('DEBUG_MODE', str(config.debug_mode)).lower() == 'true'
    
    return config

def save_config_to_file(config: SystemConfig, file_path: str):
    """Save configuration to JSON file"""
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    with open(file_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config_from_file(file_path: str) -> SystemConfig:
    """Load configuration from JSON file"""
    import json
    
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    
    # Reconstruct config object
    config = SystemConfig()
    
    # Update with loaded values
    if 'model' in config_dict:
        for key, value in config_dict['model'].items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
    
    if 'document' in config_dict:
        for key, value in config_dict['document'].items():
            if hasattr(config.document, key):
                setattr(config.document, key, value)
    
    if 'vector_store' in config_dict:
        for key, value in config_dict['vector_store'].items():
            if hasattr(config.vector_store, key):
                setattr(config.vector_store, key, value)
    
    if 'agent' in config_dict:
        for key, value in config_dict['agent'].items():
            if hasattr(config.agent, key):
                setattr(config.agent, key, value)
    
    # Update system level configs
    system_keys = ['dataset_path', 'log_level', 'debug_mode']
    for key in system_keys:
        if key in config_dict:
            setattr(config, key, config_dict[key])
    
    return config
