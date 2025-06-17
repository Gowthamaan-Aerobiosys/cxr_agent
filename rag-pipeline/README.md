# CXR Agent - Agentic RAG Pipeline

A sophisticated Retrieval-Augmented Generation (RAG) system built with QWEN 2.5 for respiratory care and mechanical ventilation assistance.

## Features

- **Agentic RAG Pipeline**: Intelligent query processing with intent analysis
- **QWEN 2.5 Integration**: State-of-the-art language model for medical domain
- **Specialized Knowledge Base**: Comprehensive respiratory care textbooks and references
- **Interactive Interfaces**: Both CLI and web-based Streamlit interface
- **Query Enhancement**: Automatic query expansion based on medical concepts
- **Source Attribution**: Detailed referencing of medical literature
- **Conversation Memory**: Persistent chat history and context

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Documents │───▶│ Document         │───▶│   Vector Store  │
│   (Medical      │    │ Processor        │    │   (ChromaDB)    │
│    Literature)  │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐            │
│  User Query     │───▶│  Intent Analysis │◀───────────┘
└─────────────────┘    └──────────────────┘
                               │
                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Response      │◀───│  QWEN 2.5 Agent │◀───│ Context         │
│   Generation    │    │                  │    │ Retrieval       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

1. **Clone the repository**:

```bash
git clone <repository-url>
cd "CXR Agent/rag-pipeline"
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Set up the environment** (optional):

```bash
# Create environment file
cp .env.example .env
# Edit .env with your preferred settings
```

## Quick Start

### 1. Initialize the System

```bash
python main.py
```

This will:

- Process all PDF documents in the dataset
- Create embeddings and populate the vector store
- Initialize the QWEN 2.5 model
- Start interactive mode

### 2. Web Interface

```bash
python main.py --gui
# or directly
streamlit run streamlit_app.py
```

### 3. Single Query

```bash
python main.py --query "What are the indications for mechanical ventilation?"
```

## Usage Examples

### Interactive CLI Mode

```bash
$ python main.py

🫁 CXR Agent - Respiratory Care Assistant
==================================================
Ask questions about mechanical ventilation, respiratory care, and pulmonary medicine.
Type 'quit', 'exit', or 'q' to end the session.

👨‍⚕️ Your question: How do I set PEEP for ARDS patients?

🔍 Analyzing your question...

🤖 CXR Agent Response:
------------------------------
For ARDS patients, PEEP should be set according to the ARDSNet protocol...
[Detailed response with clinical guidelines]

📊 Analysis:
   Question Type: Clinical Decision
   Urgency Level: Medium
   Concepts: Ventilation, Pathology

📚 Sources Referenced:
   1. ARDS Network Protocol (Page 145) - 95% relevant
   2. Mechanical Ventilation Principles (Page 67) - 87% relevant
```

### Web Interface Features

- **Quick Query Selection**: Pre-defined common questions
- **Real-time Response**: Streaming responses with progress indicators
- **Source Attribution**: Clickable references to original documents
- **Conversation History**: Persistent chat sessions
- **Query Analysis**: Visual breakdown of question intent and concepts

## Configuration

### Environment Variables

```bash
# Model Configuration
QWEN_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
LOAD_IN_4BIT=true
MAX_NEW_TOKENS=2048

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Vector Store
COLLECTION_NAME=respiratory_care_docs
EMBEDDING_MODEL=all-MiniLM-L6-v2

# System
DATASET_PATH=../dataset/books
LOG_LEVEL=INFO
DEBUG_MODE=false
```

### Configuration File

```python
from config import SystemConfig, load_config_from_file

# Load custom configuration
config = load_config_from_file('custom_config.json')
```

## Dataset Structure

The system expects medical literature in PDF format:

```
dataset/books/
├── Clinical Application Of Mechanical Ventilation - 4th Edition.pdf
├── Egans fundamentals of respiratory care 12th Edition.pdf
├── Equipment for respiratory care 2nd Edition.pdf
├── ERS Handbook of respiratory medicine 3rd edition.pdf
├── Essentials of Mechanical Ventilation.pdf
├── Oxford handbook of respiratory medicine 4th edition.pdf
├── Pilbeams mechanical ventilation physiological and clinical applications 8th edition.pdf
├── Principles of pharmacology for respiratory care 3rd edition.pdf
├── Respiratory care calculations revised 4th edition.pdf
├── Wests Pulmonary Pathophysiology - 9th edition.pdf
└── ... (additional medical references)
```

## Advanced Features

### Intent Analysis

The system automatically analyzes queries to determine:

- **Question Type**: Procedural, explanatory, clinical decision, factual, troubleshooting
- **Medical Concepts**: Ventilation, pathology, procedures, physiology
- **Urgency Level**: High, medium, low priority

### Query Enhancement

Automatically expands queries with relevant medical terminology:

```python
Original: "How to set PEEP?"
Enhanced: "How to set PEEP? mechanical ventilation respiratory mechanics ventilator settings"
```

### Agentic Behavior

- **Context-Aware Responses**: Maintains conversation context
- **Source Verification**: Cross-references multiple medical sources
- **Safety Prioritization**: Always emphasizes patient safety
- **Clinical Guidelines**: References evidence-based protocols

## API Reference

### DocumentProcessor

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
chunks = processor.process_documents("path/to/pdfs")
```

### VectorStore

```python
from document_processor import VectorStore

store = VectorStore(collection_name="medical_docs")
store.add_documents(chunks)
results = store.search("mechanical ventilation", n_results=5)
```

### QwenAgent

```python
from qwen_agent import QwenAgent, AgenticRAG

agent = QwenAgent(model_name="Qwen/Qwen2.5-7B-Instruct")
rag_system = AgenticRAG(vector_store, agent)
response = rag_system.process_query("Your medical question")
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Or run individual test modules:

```bash
python tests/test_rag_pipeline.py
```

## Performance Optimization

### Memory Usage

- **4-bit Quantization**: Reduces model memory by ~75%
- **Efficient Chunking**: Optimized text segmentation
- **Vector Caching**: Persistent embeddings storage

### Speed Optimization

- **GPU Acceleration**: Automatic CUDA detection
- **Batch Processing**: Efficient document processing
- **Streaming Responses**: Real-time response generation

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:

```bash
# Use smaller model or enable 4-bit quantization
python main.py --model Qwen/Qwen2.5-3B-Instruct
```

2. **No Documents Found**:

```bash
# Check dataset path
python main.py --dataset /path/to/your/pdfs --reprocess
```

3. **Slow Response Times**:

```bash
# Reduce max tokens or enable quantization
export MAX_NEW_TOKENS=1024
export LOAD_IN_4BIT=true
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## Medical Disclaimer

⚠️ **IMPORTANT**: This AI assistant provides educational information only. Always consult with qualified healthcare professionals for patient-specific medical decisions. The system is designed to support clinical decision-making, not replace professional medical judgment.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- QWEN 2.5 model by Alibaba Cloud
- Medical textbooks and clinical references
- Open-source libraries: transformers, langchain, chromadb, streamlit
