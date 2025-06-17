# 🫁 CXR Agent - Complete System Guide

## 🎯 Quick Start

### 1. Setup (One-time)

```bash
# Install dependencies and setup system
python setup_system.py

# Setup RAG pipeline (if needed)
cd rag-pipeline
setup.bat
```

### 2. Run Chat Interface (Recommended)

```bash
# Start the interactive chat interface
python run_chat.py

# Open browser to: http://localhost:8501
```

### 3. Explore Capabilities

```bash
# Run comprehensive demo
python demo.py
```

## 🚀 Usage Modes

### 💬 Interactive Chat Interface

**Best for**: Interactive analysis, learning, clinical decision support

```bash
python run_chat.py
```

**Features**:

- Upload CXR images via drag-and-drop
- Ask medical questions in natural language
- Get combined analysis + evidence-based interpretations
- Real-time visualizations and reports
- Three modes: Hybrid, RAG-only, Analysis-only

**Example Workflows**:

1. **Image Analysis**: Upload CXR → "Analyze this image" → Get classification, segmentation, pathology detection
2. **Medical Q&A**: "What are signs of pneumonia on chest X-ray?" → Get evidence-based answer
3. **Hybrid**: Upload CXR with pneumonia → "Explain these findings and treatment options" → Combined analysis + clinical guidance

### 🖥️ Command Line Interface

**Best for**: Batch processing, scripting, automation

```bash
# Analyze single image
python cxr_cli.py analyze sample_cxr.jpg

# Batch analysis
python cxr_cli.py batch-analyze /path/to/images/

# Ask medical questions
python cxr_cli.py query "What are the ventilation strategies for ARDS?"

# System status
python cxr_cli.py status
```

### 🌐 API Server (MCP)

**Best for**: Integration with other systems, scalable deployments

```bash
# Start API server
python mcp_server.py

# Available at: http://localhost:8000
# API docs at: http://localhost:8000/docs
```

**API Endpoints**:

- `POST /analyze` - Analyze single CXR image
- `POST /batch-analyze` - Batch analysis
- `POST /query` - RAG-based medical queries
- `GET /status` - System status
- `GET /health` - Health check

## 🔧 System Components

### 1. **CXR Agent Core** (`cxr_agent.py`)

- Main orchestrator combining all capabilities
- Async analysis and RAG queries
- Clinical report generation

### 2. **Lung Tools Package** (`lung_tools/`)

- `CXRImageProcessor`: Image preprocessing and enhancement
- `CXRClassifier`: Multi-pathology classification
- `LungSegmenter`: Lung region segmentation
- `CXRFeatureExtractor`: Feature extraction from segments
- `PathologyDetector`: Rule-based + AI pathology detection

### 3. **RAG Pipeline** (`rag-pipeline/`)

- Medical knowledge base from respiratory care textbooks
- Vector search and retrieval
- Contextual question answering

### 4. **Chat Interface** (`chat_interface.py`)

- Streamlit-based interactive UI
- File upload and real-time analysis
- Rich visualizations and reports

### 5. **MCP Server** (`mcp_server.py`)

- FastAPI-based REST API
- Scalable async processing
- Docker-ready deployment

## 📊 Analysis Capabilities

### Image Analysis

- **Preprocessing**: Enhancement, normalization, resize
- **Classification**: 14+ pathology classes with confidence scores
- **Segmentation**: Lung boundary detection and masking
- **Feature Extraction**: Morphological and intensity features
- **Pathology Detection**: Combined rule-based and AI detection
- **Reporting**: Structured clinical reports

### RAG Capabilities

- **Knowledge Base**: 18 respiratory care textbooks
- **Query Types**: Diagnostic criteria, treatment protocols, physiology
- **Context-Aware**: Can incorporate analysis results
- **Evidence-Based**: Provides source citations

## 🎨 Chat Interface Features

### Chat Modes

1. **🔄 Hybrid Mode** (Default)

   - Combines image analysis with medical knowledge
   - Best for clinical decision support

2. **💬 RAG-Only Mode**

   - Pure question-answering from medical literature
   - Best for study and reference

3. **🔍 Analysis-Only Mode**
   - Focus on technical image analysis
   - Best for radiology workflow

### Interactive Features

- **Drag-and-drop image upload**
- **Real-time analysis progress**
- **Interactive charts and visualizations**
- **Clinical report generation**
- **Sample questions for quick start**
- **System status monitoring**
- **Analysis history and summary**

## 🐳 Deployment Options

### Local Development

```bash
python run_chat.py          # Streamlit interface
python mcp_server.py         # API server
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t cxr-agent .
docker run -p 8000:8000 -p 8501:8501 cxr-agent
```

### Production Deployment

- Use `docker-compose.yml` for orchestration
- Configure environment variables
- Setup reverse proxy (nginx) if needed
- Consider scaling with Kubernetes

## 🔍 Example Use Cases

### 1. **Emergency Department**

```
Workflow: Upload CXR → "Is this pneumothorax?" → Get immediate analysis + treatment protocols
```

### 2. **Medical Education**

```
Workflow: "What are the radiological signs of ARDS?" → Get comprehensive educational content
```

### 3. **Research**

```
Workflow: Batch analyze 100+ images → Export results → Statistical analysis
```

### 4. **Teleradiology**

```
Workflow: API integration → Automated initial screening → Radiologist review
```

## 🛠️ Troubleshooting

### Common Issues

**Setup Problems**:

```bash
# Check Python version (needs 3.8+)
python --version

# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Setup RAG pipeline
cd rag-pipeline && setup.bat
```

**Memory Issues**:

- Models require ~8GB RAM
- Use smaller batch sizes
- Consider GPU acceleration if available

**Import Errors**:

```bash
# Check specific package
python -c "import torch; print(torch.__version__)"

# Install missing packages
pip install torch torchvision transformers
```

### Performance Optimization

**For CPU-only systems**:

- Models will run slower but still functional
- Consider using smaller model variants
- Reduce batch sizes for analysis

**For GPU systems**:

- Install CUDA-compatible PyTorch
- Enable GPU acceleration in config
- Monitor GPU memory usage

## 📈 System Monitoring

### Built-in Status Checks

```bash
# CLI status check
python cxr_cli.py status

# API health check
curl http://localhost:8000/health

# Chat interface: Click "System Status" in sidebar
```

### Logs and Debugging

- All components use Python logging
- Check console output for errors
- Enable debug mode in config files

## 🔮 Future Enhancements

### Planned Features

- **Multi-modal analysis**: CT scans, MRI support
- **DICOM integration**: Direct PACS connectivity
- **Advanced AI models**: State-of-the-art vision transformers
- **Clinical workflow**: Integration with EMR systems
- **Mobile app**: Smartphone-based analysis

### Customization Options

- **Custom models**: Train on your datasets
- **Domain-specific knowledge**: Add specialized textbooks
- **Workflow integration**: Custom API endpoints
- **UI themes**: Customize chat interface appearance

---

## 📞 Support

For issues, questions, or contributions:

1. Check this guide and README.md
2. Run the demo to test all features
3. Check logs for specific error messages
4. Review the code for customization options

**Happy analyzing! 🫁✨**
