# CXR Agent MCP Server Documentation

## Architecture

### Core Components

1. **MCP Server** (`mcp_server.py`)

   - Main server implementation
   - Tool registration and execution
   - Request routing and response handling

2. **Model Registry** (`model_registry.py`)

   - Centralized model management
   - Lazy loading and caching
   - Memory management and cleanup

3. **Model Adapters** (`model_adapters.py`)

   - Unified interface for different model types
   - Base adapter class for consistency
   - Specific adapters for each model type

4. **Configuration** (`config/mcp_config.json`)
   - Model settings and paths
   - Server configuration
   - Preprocessing parameters

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Client                              │
│            (Python/JavaScript/Other Language)                │
└────────────────────────┬────────────────────────────────────┘
                         │ MCP Protocol
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    MCP Server                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Tool Registration                        │  │
│  │  • classify_cxr_binary                               │  │
│  │  • classify_cxr_diseases                             │  │
│  │  • segment_lungs                                     │  │
│  │  • query_medical_knowledge                           │  │
│  │  • generate_cxr_report                               │  │
│  │  • analyze_cxr_complete                              │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐  │
│  │              Model Registry                           │  │
│  │  • Load/Unload Models                                │  │
│  │  • Model Caching                                     │  │
│  │  • Memory Management                                 │  │
│  └──────────────────────┬───────────────────────────────┘  │
└─────────────────────────┼────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
│   Binary     │  │  14-Class   │  │    Lung     │
│ Classifier   │  │ Classifier  │  │ Segmentation│
└──────────────┘  └─────────────┘  └─────────────┘
        │                 │                 │
┌───────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
│     RAG      │  │   Report    │  │  Feature    │
│    Agent     │  │  Generator  │  │  Extractor  │
└──────────────┘  └─────────────┘  └─────────────┘
```

## Available Models

### 1. Binary Classifier

- **Purpose**: Classify CXR as Normal or Abnormal
- **Model**: Swin Transformer (configurable)
- **Input**: Chest X-ray image
- **Output**: Binary classification with probabilities

### 2. Multi-Class Classifier (14 Diseases)

- **Purpose**: Detect 14 different pathologies
- **Model**: DenseNet-121 (configurable)
- **Diseases**:
  - Atelectasis, Cardiomegaly, Effusion, Infiltration
  - Mass, Nodule, Pneumonia, Pneumothorax
  - Consolidation, Edema, Emphysema, Fibrosis
  - Pleural Thickening, Hernia
- **Input**: Chest X-ray image
- **Output**: Probability score for each disease

### 3. Lung Segmentation

- **Purpose**: Segment lung regions
- **Model**: U-Net (configurable)
- **Input**: Chest X-ray image
- **Output**: Segmentation mask and metrics

### 4. RAG System

- **Purpose**: Answer medical questions using medical literature
- **Model**: DeepSeek-R1 / Qwen (configurable)
- **Input**: Medical query
- **Output**: Answer with sources

### 5. Report Generator

- **Purpose**: Generate radiology reports
- **Model**: BioGPT (configurable)
- **Input**: CXR image + findings
- **Output**: Structured radiology report

### 6. Feature Extractor

- **Purpose**: Extract quantitative radiomics features
- **Model**: PyRadiomics-based
- **Input**: CXR image + optional mask
- **Output**: Feature vector

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Install Dependencies

```bash
# Navigate to project directory
cd "CXR Agent"

# Create virtual environment (recommended)
python -m venv cenv
source cenv/bin/activate  # On Windows: cenv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Download Model Weights

```bash
# Create weights directory
mkdir -p weights

# Download models (example URLs - replace with actual)
# Binary classifier
wget https://example.com/binary_classifier.pth -O weights/binary_classifier.pth

# 14-class classifier
wget https://example.com/fourteen_class_classifier.pth -O weights/fourteen_class_classifier

# Segmentation model
wget https://example.com/segmentation_model.pth -O weights/segmentation_model.pth
```

### Prepare RAG Documents

```bash
# Place medical literature PDFs in dataset/books/
cp /path/to/medical/pdfs/* dataset/books/

# Initialize vector database (one-time setup)
cd rag-pipeline
python document_processor.py --initialize
```

## Configuration

### Config File Structure

Edit `config/mcp_config.json`:

```json
{
  "models": {
    "binary_classifier": {
      "enabled": true,
      "checkpoint_path": "weights/binary_classifier.pth",
      "model_type": "swin_transformer"
    },
    "multiclass_classifier": {
      "enabled": true,
      "checkpoint_path": "weights/fourteen_class_classifier",
      "num_classes": 14,
      "model_type": "densenet121"
    }
  },
  "server_config": {
    "device": "cuda",
    "cache_models": true,
    "max_batch_size": 4
  }
}
```

### Key Configuration Options

- **device**: "cuda" or "cpu"
- **cache_models**: Keep models in memory (true) or load on-demand (false)
- **checkpoint_path**: Path to model weights
- **enabled**: Enable/disable specific models

## Running the Server

### Start the MCP Server

```bash
# Default configuration
python mcp_server.py

# Custom configuration
python mcp_server.py --config config/mcp_config.json
```

### Server Logs

The server will log:

- Model loading status
- Inference requests
- Errors and warnings
- Performance metrics

## Using the Client

### Python Client

```python
import asyncio
from mcp_client_example import CXRAgentClient

async def analyze_xray():
    # Initialize client
    client = CXRAgentClient()
    await client.connect()

    # Binary classification
    result = await client.classify_binary("path/to/xray.jpg")
    print(f"Prediction: {result['prediction']}")

    # Disease detection
    diseases = await client.classify_diseases("path/to/xray.jpg")
    print(f"Detected diseases: {diseases['detected_diseases']}")

    # Complete analysis
    complete = await client.analyze_complete("path/to/xray.jpg")
    print(complete)

asyncio.run(analyze_xray())
```

### Available Client Methods

1. **classify_binary(image_path, threshold)**

   - Binary classification

2. **classify_diseases(image_path, threshold, top_k)**

   - Multi-class disease detection

3. **segment_lungs(image_path, save_mask, output_path)**

   - Lung segmentation

4. **query_medical_knowledge(query, top_k, include_sources)**

   - RAG medical Q&A

5. **generate_report(image_path, findings, clinical_context, report_style)**

   - Generate radiology report

6. **analyze_complete(image_path, include_segmentation, include_report, clinical_context)**

   - Comprehensive analysis

7. **list_models()**

   - List available models

8. **load_model(model_name)**

   - Load specific model

9. **unload_model(model_name)**
   - Unload model to free memory

## MCP Tools Reference

### Tool: classify_cxr_binary

Classify chest X-ray as Normal or Abnormal.

**Input Schema:**

```json
{
  "image_path": "string (required)",
  "threshold": "number (optional, default: 0.5)"
}
```

**Output:**

```json
{
  "prediction": "Normal | Abnormal",
  "confidence": 0.95,
  "probabilities": {
    "Normal": 0.05,
    "Abnormal": 0.95
  },
  "inference_time_ms": 150
}
```

### Tool: classify_cxr_diseases

Detect 14 different pathologies.

**Input Schema:**

```json
{
  "image_path": "string (required)",
  "threshold": "number (optional, default: 0.3)",
  "top_k": "integer (optional, default: 5)"
}
```

**Output:**

```json
{
  "detected_diseases": {
    "Pneumonia": 0.85,
    "Infiltration": 0.72
  },
  "top_predictions": {
    "Pneumonia": 0.85,
    "Infiltration": 0.72,
    "Consolidation": 0.45
  },
  "num_detected": 2,
  "inference_time_ms": 200
}
```

### Tool: segment_lungs

Segment lung regions.

**Input Schema:**

```json
{
  "image_path": "string (required)",
  "save_mask": "boolean (optional, default: false)",
  "output_path": "string (optional)"
}
```

**Output:**

```json
{
  "lung_area_pixels": 125000,
  "lung_ratio": 0.45,
  "left_lung_area": 62000,
  "right_lung_area": 63000,
  "mask_saved": true,
  "output_path": "outputs/mask.png"
}
```

### Tool: query_medical_knowledge

Query medical knowledge base using RAG.

**Input Schema:**

```json
{
  "query": "string (required)",
  "top_k_docs": "integer (optional, default: 5)",
  "include_sources": "boolean (optional, default: true)"
}
```

**Output:**

```json
{
  "query": "What are signs of pneumonia?",
  "answer": "Pneumonia on chest X-ray typically shows...",
  "sources": [{ "title": "Respiratory Care Handbook", "page": 45 }],
  "query_time_ms": 500
}
```

### Tool: analyze_cxr_complete

Comprehensive CXR analysis.

**Input Schema:**

```json
{
  "image_path": "string (required)",
  "include_segmentation": "boolean (optional, default: true)",
  "include_report": "boolean (optional, default: false)",
  "clinical_context": "string (optional)"
}
```

**Output:**

```json
{
  "binary_classification": {...},
  "disease_classification": {...},
  "segmentation": {...},
  "report": {...}
}
```

## Adding New Models

### Step 1: Create Model Adapter

Create a new adapter in `model_adapters.py`:

```python
class NewModelAdapter(BaseModelAdapter):
    def __init__(self, checkpoint_path, device=None):
        super().__init__(device)
        self.checkpoint_path = checkpoint_path

    async def load(self):
        # Load your model
        self.model = load_your_model(self.checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    async def predict(self, image_path, **kwargs):
        # Run inference
        result = self.model(preprocess(image_path))
        return {"result": result}
```

### Step 2: Register in Model Registry

Add to `ModelType` enum in `model_registry.py`:

```python
class ModelType(Enum):
    # ... existing types
    NEW_MODEL = "new_model"
```

Add loading logic in `ModelRegistry._load_model()`:

```python
elif model_type == ModelType.NEW_MODEL:
    adapter = NewModelAdapter(
        checkpoint_path=model_config.get("checkpoint_path"),
        device=self.device
    )
```

### Step 3: Add MCP Tool

Add tool in `mcp_server.py`:

```python
@self.server.list_tools()
async def list_tools() -> list[Tool]:
    tools.append(Tool(
        name="new_model_tool",
        description="Description of what this tool does",
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                # ... other parameters
            },
            "required": ["image_path"]
        }
    ))
```

Add tool handler:

```python
@self.server.call_tool()
async def call_tool(name: str, arguments: Any):
    if name == "new_model_tool":
        result = await self._run_new_model(arguments)
```

### Step 4: Update Configuration

Add to `config/mcp_config.json`:

```json
{
  "models": {
    "new_model": {
      "enabled": true,
      "checkpoint_path": "weights/new_model.pth",
      "model_type": "custom_architecture"
    }
  }
}
```

## Performance Optimization

### Memory Management

```python
# Disable model caching to reduce memory
{
  "server_config": {
    "cache_models": false
  }
}

# Manually unload models
await client.unload_model("binary_classifier")
```

### GPU Optimization

```python
# Use mixed precision
torch.set_float32_matmul_precision('medium')

# Enable TF32 on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
```

### Batch Processing

For multiple images, batch them when possible:

```python
# Process multiple images efficiently
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = await asyncio.gather(*[
    client.classify_binary(img) for img in images
])
```

## Troubleshooting

### Model Loading Errors

**Error**: "Checkpoint not found"

- **Solution**: Verify checkpoint path in config
- Check file exists: `ls weights/`

**Error**: "CUDA out of memory"

- **Solution**:
  - Use CPU: Set `"device": "cpu"` in config
  - Reduce batch size
  - Enable model unloading: `"cache_models": false`

### Connection Errors

**Error**: "Cannot connect to MCP server"

- **Solution**:
  - Verify server is running
  - Check server logs for errors
  - Ensure correct server script path

### Inference Errors

**Error**: "Image format not supported"

- **Solution**: Convert to JPEG/PNG
- Use PIL: `Image.open(path).convert('RGB').save('converted.jpg')`

## API Integration

### REST API Wrapper

Create a REST API wrapper (example with FastAPI):

```python
from fastapi import FastAPI, UploadFile
from mcp_client_example import CXRAgentClient

app = FastAPI()
client = CXRAgentClient()

@app.post("/analyze")
async def analyze(file: UploadFile):
    # Save uploaded file
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    # Analyze with MCP
    result = await client.analyze_complete(path)
    return result
```

### WebSocket Streaming

For real-time updates:

```python
@app.websocket("/ws/analyze")
async def analyze_stream(websocket: WebSocket):
    await websocket.accept()

    # Stream progress updates
    await websocket.send_json({"status": "loading_model"})
    # ... analysis
    await websocket.send_json({"status": "complete", "result": result})
```

## Security Considerations

1. **Input Validation**: Always validate image paths and parameters
2. **Authentication**: Add authentication for production deployments
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Resource Limits**: Set memory and CPU limits
5. **HIPAA Compliance**: Ensure PHI handling compliance if applicable

## License

See LICENSE file for details.

## Support

For issues, questions, or contributions:

- GitHub Issues: [link]
- Documentation: [link]
- Email: [contact]
