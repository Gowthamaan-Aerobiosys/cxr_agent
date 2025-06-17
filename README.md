# CXR Agent - Comprehensive Chest X-Ray Analysis System

A state-of-the-art AI system that combines advanced computer vision techniques with agentic Retrieval-Augmented Generation (RAG) for comprehensive chest X-ray analysis.

## 🎯 Features

### 🔬 Advanced Medical Image Analysis

- **Multi-task Classification**: Detect 14+ pathologies including pneumonia, pneumothorax, cardiomegaly, etc.
- **Lung Segmentation**: Precise segmentation of lung fields and anatomical structures
- **Feature Extraction**: Comprehensive radiological and morphological feature analysis
- **Pathology Detection**: Rule-based and AI-powered detection of specific conditions

### � Agentic RAG System

- **Medical Knowledge Base**: Integration with extensive respiratory care literature
- **Intelligent Query Processing**: Context-aware medical question answering
- **Clinical Interpretation**: AI-powered interpretation of imaging findings
- **Evidence-Based Recommendations**: Treatment suggestions based on current medical literature

### 🌐 Scalable Architecture

- **MCP Server**: Model Context Protocol server for enterprise integration
- **REST API**: Full-featured API for programmatic access
- **CLI Interface**: Command-line tools for batch processing and automation
- **Web Interface**: User-friendly web dashboard (via Streamlit)

## � Project Structure

```
CXR Agent/
├── lung_tools/                    # Core image analysis modules
│   ├── __init__.py
│   ├── image_processor.py         # Image preprocessing and utilities
│   ├── classifier.py              # Multi-pathology classification
│   ├── segmentation.py            # Lung segmentation
│   ├── feature_extractor.py       # Comprehensive feature extraction
│   └── pathology_detector.py      # Advanced pathology detection
├── rag-pipeline/                  # Agentic RAG system
│   ├── config.py                  # Configuration management
│   ├── qwen_agent.py             # QWEN model integration
│   ├── document_processor.py      # Document processing
│   ├── main.py                   # RAG pipeline main
│   └── streamlit_app.py          # Web interface
├── dataset/books/                 # Medical literature corpus
├── cxr_agent.py                  # Main CXR Agent integration
├── mcp_server.py                 # MCP server implementation
├── cxr_cli.py                    # Command-line interface
├── setup.py                      # Setup and installation script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ storage space

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd "CXR Agent"
```

2. **Run the setup script**

```bash
python setup.py
```

3. **Configure the system**

```bash
cp .env.template .env
# Edit .env with your configurations
```

4. **Place medical documents**
   Place your medical literature PDF files in `dataset/books/`

5. **Initialize the RAG system**
   python main.py --query "What are PEEP settings for ARDS?"

````

### 3. Web Interface

Visit `http://localhost:8501` after running the GUI mode for an interactive chat interface with:

- Real-time responses from medical literature
- Source attribution and relevance scoring
- Query analysis and concept detection
- Conversation history

## Architecture

The system implements an agentic RAG (Retrieval-Augmented Generation) pipeline:

1. **Document Processing**: Extracts and chunks text from medical PDFs
2. **Vector Storage**: Creates searchable embeddings using ChromaDB
3. **Intent Analysis**: Analyzes queries for medical concepts and urgency
4. **Context Retrieval**: Finds relevant passages from medical literature
5. **Response Generation**: Uses QWEN 2.5 to generate evidence-based answers

## Medical Disclaimer

⚠️ **IMPORTANT**: This AI assistant provides educational information only. Always consult with qualified healthcare professionals for patient-specific medical decisions.

## 📚 Documentation

Comprehensive documentation is available in the `/docs` directory, including:

- Installation guide
- Configuration details
- API documentation
- Usage examples
- Troubleshooting tips

## 🚧 Troubleshooting

Common issues and solutions:

- **Installation errors**: Ensure all prerequisites are met, and follow the installation guide closely.
- **CUDA errors**: Check CUDA installation and compatibility with your GPU.
- **API not starting**: Ensure no other services are using the same port, and check the logs for errors.
- **Model loading issues**: Verify the model path and permissions.

For unresolved issues, please create an issue on GitHub with detailed information about your problem.

## 💻 Usage

### Command Line Interface

#### Analyze a single CXR image
```bash
python cxr_cli.py analyze path/to/image.jpg -o results/
````

#### Batch analysis

```bash
python cxr_cli.py batch path/to/images/ -o results/
```

#### Query medical knowledge

```bash
python cxr_cli.py query "What are the signs of pneumonia on chest X-ray?"
```

#### Check system status

```bash
python cxr_cli.py status
```

### MCP Server (API)

#### Start the server

```bash
python mcp_server.py --host 0.0.0.0 --port 8000
```

#### API Endpoints

- `POST /analyze` - Analyze single CXR image
- `POST /batch_analyze` - Batch analysis
- `POST /upload_analyze` - Upload and analyze
- `POST /query` - Medical knowledge queries
- `GET /status` - System status
- `GET /health` - Health check

#### Example API usage

```python
import requests

# Analyze image
response = requests.post('http://localhost:8000/analyze', json={
    'image_path': 'path/to/image.jpg',
    'include_rag': True,
    'generate_report': True
})

results = response.json()
```

### Web Interface

Start the Streamlit web interface:

```bash
cd rag-pipeline
streamlit run streamlit_app.py
```

Access at `http://localhost:8501`

## 🔧 Configuration

### Environment Variables

```bash
# Model Configuration
QWEN_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
LOAD_IN_4BIT=true
MAX_NEW_TOKENS=2048

# Vector Store
COLLECTION_NAME=respiratory_care_docs
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
```

### Configuration File

Edit `configs/default_config.json` for detailed configuration:

```json
{
  "model": {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "load_in_4bit": true,
    "max_new_tokens": 2048,
    "temperature": 0.7
  },
  "document": {
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "vector_store": {
    "collection_name": "respiratory_care_docs",
    "embedding_model": "all-MiniLM-L6-v2"
  }
}
```

## 🏗️ Architecture

### Core Components

1. **Image Processing Pipeline**

   - Preprocessing and normalization
   - Enhancement and noise reduction
   - ROI extraction and preparation

2. **AI Analysis Engine**

   - DenseNet-121 based classification
   - U-Net segmentation
   - Multi-modal pathology detection

3. **Agentic RAG System**

   - QWEN 2.5 language model
   - ChromaDB vector store
   - Intelligent query processing

4. **MCP Server**
   - FastAPI-based REST API
   - Async processing
   - Batch analysis support

### Data Flow

```
CXR Image → Preprocessing → Classification ↘
                        → Segmentation    → Feature Extraction → Pathology Detection
                        → Enhancement    ↗                            ↓
                                                              RAG Analysis → Clinical Report
```

## 📊 Supported Pathologies

- Atelectasis
- Cardiomegaly
- Effusion (Pleural)
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema (Pulmonary)
- Emphysema
- Fibrosis
- Pleural Thickening
- Hernia

## 🎯 Performance

### Classification Metrics

- Multi-label AUC: 0.85+ (target)
- Sensitivity: 0.80+ per pathology
- Specificity: 0.90+ per pathology

### Processing Speed

- Single image analysis: ~30-60 seconds
- Batch processing: ~1000 images/hour
- Real-time API response: <2 minutes

### Resource Requirements

- GPU: 8GB+ VRAM (recommended)
- CPU: 16GB+ RAM
- Storage: 2GB+ per 1000 processed images

## 🔒 Security & Compliance

- **Data Privacy**: No data persistence by default
- **HIPAA Consideration**: Configurable for healthcare compliance
- **Audit Logging**: Comprehensive analysis tracking
- **API Security**: Token-based authentication support

## 🧪 Testing

Run the test suite:

```bash
cd rag-pipeline
python -m pytest tests/ -v
```

## 📝 Clinical Disclaimer

⚠️ **IMPORTANT MEDICAL DISCLAIMER**

This system is designed for **educational and research purposes only**. It is **NOT intended for clinical diagnosis or treatment decisions**.

- All AI-generated findings must be validated by qualified radiologists
- Clinical correlation with patient history is essential
- This tool should not replace professional medical judgment
- Not approved for clinical use without proper validation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
pip install -r requirements.txt
pip install -e .
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Medical literature corpus from respiratory care textbooks
- QWEN 2.5 model by Alibaba Cloud
- Open-source medical imaging libraries
- ChromaDB for vector storage
- FastAPI for web framework

## 📞 Support

For technical support or questions:

- Create an issue on GitHub
- Check the documentation in `/docs`
- Review the FAQ section

## 🔄 Version History

### v1.0.0 (Current)

- Initial release
- Complete CXR analysis pipeline
- Agentic RAG integration
- MCP server implementation
- CLI and web interfaces

## 🚧 Roadmap

### Short Term

- [ ] Enhanced pathology detection algorithms
- [ ] Additional imaging modalities support
- [ ] Performance optimizations
- [ ] Extended test coverage

### Long Term

- [ ] 3D imaging support (CT integration)
- [ ] Multi-language support
- [ ] Mobile application
- [ ] Cloud deployment options
- [ ] Integration with PACS systems

---

**Built with ❤️ for advancing medical AI and improving patient care**
