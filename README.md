# CXR Agent - AI-Powered Chest X-ray Analysis System

## 🎯 Overview

**CXR Agent** provides a single, conversational interface where users interact with one LLM that intelligently orchestrates all capabilities. Users can upload chest X-rays, ask questions, or do both - the LLM automatically determines what's needed and coordinates the appropriate models.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd CXR-Agent

# Install dependencies
pip install -r requirements.txt
```

### Running the Agent

**Option 1: Web Interface (Recommended)**

```bash
python main.py --ui
# or simply
python main.py
```

This starts the Streamlit web interface at http://localhost:8501

**Option 2: Interactive CLI**

```bash
python main.py --cli
```

This starts an interactive terminal session where you can chat with the agent.

**Option 3: MCP Server**

```bash
python main.py --mcp
```

This starts the Model Context Protocol server for integration with other tools.

**Option 4: Demo Mode**

```bash
python main.py --demo
```

This runs example queries to demonstrate the agent's capabilities.

## 📁 Project Structure

```
CXR Agent/
├── main.py                      # Main entry point - run this!
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── agent.py                 # Unified Agent orchestrator
│   │
│   ├── models/                  # AI model adapters
│   │   ├── __init__.py
│   │   ├── adapters.py          # Model adapter implementations
│   │   └── registry.py          # Model registry and management
│   │
│   ├── rag/                     # RAG pipeline for medical Q&A
│   │   ├── __init__.py
│   │   ├── llm_engine.py        # LLM interface (DeepSeek, etc.)
│   │   ├── document_processor.py # PDF processing & vectorization
│   │   ├── config.py            # RAG configuration
│   │   └── utils.py             # Utility functions
│   │
│   ├── web/                     # Web interface
│   │   ├── __init__.py
│   │   └── app.py               # Streamlit application
│   │
│   └── servers/                 # Server implementations
│       ├── __init__.py
│       └── mcp_server.py        # Model Context Protocol server
│
├── config/                      # Configuration files
│   ├── mcp_config.json          # MCP server configuration
│   └── mcp_config.template.json # Template for configuration
│
├── examples/                    # Example scripts
│   └── basic_usage.py           # Programmatic usage examples
│
├── weights/                     # Model weights (download separately)
│   ├── binary_classifier.pth
│   └── fourteen_class_classifier/
│
├── dataset/                     # Medical literature
│   └── books/                   # PDF medical textbooks
│
└── rag_pipeline/                # Legacy RAG data (kept for compatibility)
    └── chroma_db/               # Vector database
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER                                  │
│          (Single Chat Interface)                         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              UNIFIED AGENT                               │
│         (LLM-Based Orchestrator)                         │
│                                                          │
│  • Analyzes user intent                                 │
│  • Determines required models                           │
│  • Coordinates execution                                │
│  • Combines results                                     │
│  • Generates unified response                           │
└─────────┬──────────────────────────┬────────────────────┘
          │                          │
          ▼                          ▼
┌──────────────────┐      ┌──────────────────┐
│  VISION MODELS   │      │   LLM + RAG      │
│                  │      │                  │
│  • Binary        │      │  • DeepSeek R1   │
│  • Multi-class   │      │  • Vector Store  │
│  • Swin Large    │      │  • Med Lit.      │
└──────────────────┘      └──────────────────┘
```

## � Usage Examples

### Web Interface

1. Start the application:

   ```bash
   python main.py --ui
   ```

2. Open your browser to http://localhost:8501

3. Interact with the agent:
   - Type questions: "What are signs of pneumonia?"
   - Upload X-rays and ask: "Is this normal?"
   - Follow up: "What's the treatment?"

### CLI Mode

```bash
python main.py --cli
```

Then interact:

```
💬 You: What are the radiological signs of pneumonia?
🤖 Agent: [Provides detailed answer with medical references]

💬 You: upload path/to/xray.jpg
🤖 Agent: [Analyzes the image]

💬 You: What diseases do you detect?
🤖 Agent: [Provides analysis based on the uploaded image]
```

### Programmatic Usage

```python
import asyncio
from src.agent import UnifiedAgent
from src.rag.llm_engine import LLMEngine
from src.rag.document_processor import VectorStore

async def analyze():
    # Initialize components
    config = {
        "models": {
            "binary_classifier": {
                "enabled": True,
                "checkpoint_path": "weights/binary_classifier.pth",
                "model_type": "swin_transformer"
            },
            "multiclass_classifier": {
                "enabled": True,
                "checkpoint_path": "weights/fourteen_class_classifier",
                "num_classes": 14,
                "model_type": "swin_transformer"
            },
            "rag": {
                "enabled": True,
                "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                "use_tgi": True
            }
        },
        "device": "cuda"
    }

    agent = UnifiedAgent(config)

    # Ask a question
    response = await agent.process_message(
        query="What are signs of pneumonia?"
    )
    print(response['answer'])

    # Analyze an image
    response = await agent.process_message(
        query="Is this X-ray normal?",
        image_path="path/to/xray.jpg"
    )
    print(response['answer'])

asyncio.run(analyze())
```

See `examples/basic_usage.py` for more examples.

## 🔑 Key Features

### 1. **Intelligent Intent Detection**

Automatically determines if user wants:

- Image analysis only
- Medical knowledge only
- Combined analysis
- Follow-up on previous image

### 2. **Lazy Model Loading**

Vision models load only when needed:

- First image query → loads models
- Subsequent questions → uses cached models
- Pure Q&A → never loads vision models

### 3. **Context Management**

Remembers conversation history and image context:

- "What about that X-ray?" → Uses previous image
- Multi-turn conversations
- Contextual follow-ups

## 🆕 What's New?

### Organized Project Structure

- ✅ Clean separation of concerns
- ✅ Modular architecture
- ✅ Easy to extend and maintain

### Single Entry Point

- ✅ One `main.py` file to run everything
- ✅ Multiple modes (UI, CLI, MCP, Demo)
- ✅ Simple command-line interface

### Better Import Management

- ✅ Proper Python package structure
- ✅ Relative imports
- ✅ No path manipulation needed

## 🚀 Quick Start (Recommended)

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the agent:**
   ```bash
   python main.py
   ```

That's it! The web interface will open at http://localhost:8501

## 💬 Example Interactions

### Example 1: Pure Medical Question

```
User: "What are the main causes of pleural effusion?"

Agent: [Searches medical literature via RAG]
       [Generates comprehensive answer with sources]
```

### Example 2: Image Analysis

```
User: [Uploads X-ray] "Is this chest X-ray normal or abnormal?"

Agent: [Runs binary classifier]
       [Runs multi-class classifier]
       [Combines results with medical knowledge]

Response: "This chest X-ray appears ABNORMAL with 87% confidence.
           The AI has detected potential Pneumonia (65%) and
           Pleural Effusion (42%). These findings suggest..."
```

### Example 3: Combined Query

```
User: [Uploads X-ray] "What treatment options are available for this?"

Agent: [Analyzes image → Detects Pneumothorax]
      [Searches medical literature for Pneumothorax treatment]
      [Generates contextualized answer]

Response: "Based on the detected Pneumothorax in your X-ray,
          treatment options typically include..."
```

### Example 4: Follow-up Questions

```
User: "What's the prognosis?"

Agent: [Uses previous image context]
      [Searches for prognosis information]

Response: "For Pneumothorax as detected in the previous image..."
```

## 🎯 How Intent Detection Works

The `UnifiedAgent` automatically analyzes user intent:

| User Input                       | Detected Intent    | Actions Taken              |
| -------------------------------- | ------------------ | -------------------------- |
| "What causes ARDS?"              | `medical_question` | → RAG search only          |
| [Image] + "Is this normal?"      | `image_analysis`   | → Binary classifier        |
| [Image] + "What diseases?"       | `image_analysis`   | → Multi-class classifier   |
| [Image] + "Analyze this"         | `image_analysis`   | → Both classifiers         |
| [Image] + "Treatment options?"   | `combined`         | → Image analysis + RAG     |
| "What about the previous image?" | `image_question`   | → Use cached context + RAG |

## 📁 New Files

### Core Components

1. **`unified_agent.py`** - Main orchestration logic

   - `UnifiedAgent` class - Central coordinator
   - Intent analysis
   - Model routing
   - Response formatting

2. **`app_unified.py`** - Streamlit UI

   - Single chat interface
   - Image upload + text input
   - Conversation history
   - Real-time responses

3. **`example_unified_agent.py`** - Usage examples
   - Programmatic API examples
   - Different query types
   - Conversation management

## 🔑 Key Features

### 1. **Intelligent Intent Detection**

Automatically determines if user wants:

- Image analysis only
- Medical knowledge only
- Combined analysis
- Follow-up on previous image

### 2. **Lazy Model Loading**

Vision models load only when needed:

```python
# First image query → loads models
# Subsequent questions → uses cached models
# Pure Q&A → never loads vision models
```

### 3. **Context Preservation**

```python
agent.process_message("Analyze this X-ray", image="scan.jpg")
# Later...
agent.process_message("What's the clinical significance?")
# ↑ Automatically uses previous image context
```

### 4. **Query Enhancement**

```python
# User asks: "What treatment is needed?"
# With detected Pneumonia in image
# Enhanced query: "What treatment is needed? Pneumonia"
# → Better RAG retrieval
```

### 5. **Unified Response Format**

```python
{
    "answer": "The LLM's natural language response",
    "thinking": "Reasoning process (DeepSeek)",
    "has_thinking": True,
    "image_analysis": {
        "binary": {...},
        "diseases": {...}
    },
    "sources": [
        {"source": "...", "page": "...", "relevance_score": 0.85}
    ],
    "intent": {...},
    "timestamp": "..."
}
```

## 🎨 UI Features

### Streamlit App (`app_unified.py`)

- **Single Chat Input** - Natural conversation flow
- **Image Upload** - Drag & drop X-rays
- **Auto-Detection** - LLM figures out what to do
- **Rich Responses** - Formatted answers with sources
- **Thinking Display** - See AI reasoning (optional)
- **Analysis Details** - Show/hide technical results
- **Quick Examples** - One-click example queries
- **Conversation History** - Full chat log

## 🔧 Configuration

Use the same `config/mcp_config.json`:

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
      "model_type": "swin_transformer"
    },
    "rag": {
      "enabled": true,
      "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
      "use_tgi": true,
      "vector_db_path": "rag_pipeline/chroma_db",
      "documents_path": "dataset/books"
    }
  },
  "device": "cuda"
}
```

## 🆚 Comparison: Old vs New

### Old Approach (`app.py`)

```
User → Tab Selection → Specific Interface → Manual Model Selection
```

### New Approach (`app_unified.py`)

```
User → Natural Question → LLM Orchestration → Automatic Routing
```

| Feature        | Old (app.py)             | New (app_unified.py)                |
| -------------- | ------------------------ | ----------------------------------- |
| Interface      | 3 separate tabs          | Single chat window                  |
| User Action    | Select tab, choose model | Just ask naturally                  |
| Image Handling | Manual upload per tab    | Upload once, ask multiple questions |
| Context        | No cross-tab context     | Full conversation context           |
| LLM Role       | Just Q&A in tab 2        | Central orchestrator for everything |
| Complexity     | User decides routing     | LLM decides routing                 |

## 📊 Use Cases

### Clinical Workflow

```
1. Upload patient's CXR
2. Ask: "Initial assessment?"
   → Binary + multi-class analysis
3. Ask: "What's the differential diagnosis?"
   → RAG search with findings
4. Ask: "Recommended follow-up?"
   → Clinical guidelines from literature
```

### Educational

```
1. Ask: "What is cardiomegaly?"
   → RAG explanation
2. Upload example: "Is this cardiomegaly?"
   → Image analysis + comparison
3. Ask: "What other conditions show similar signs?"
   → RAG search with context
```

### Research

```
1. Batch process multiple X-rays
2. Ask questions about each
3. Compare findings
4. Get literature references
```

## 🔬 Technical Details

### Intent Analysis Algorithm

```python
def _analyze_user_intent(query, has_image):
    # Keywords detection
    if has_image:
        if "normal" or "abnormal" → binary classifier
        if "disease" or "detect" → multiclass classifier
        if general question → combined analysis
    else:
        if refers to "this" or "it" → use cached image
        else → pure RAG query
```

### Model Loading Strategy

```python
# Lazy loading - only when needed
if intent.requires_image_analysis:
    if not binary_classifier:
        load_binary_classifier()
    if not multiclass_classifier:
        load_multiclass_classifier()

# Models stay in memory for subsequent requests
```

### Response Generation Pipeline

```
1. Analyze Intent
2. Execute Required Models
   - Image analysis (if needed)
   - RAG retrieval (if needed)
3. Format Context for LLM
4. Generate Unified Prompt
5. LLM Response Generation
6. Parse & Format Response
7. Update Conversation History
```

## 🐛 Troubleshooting

### Issue: "Models not loading"

```bash
# Check model paths in config
# Ensure weights exist:
ls weights/
# Should show:
# - binary_classifier.pth
# - fourteen_class_classifier/
```

### Issue: "RAG not working"

```bash
# Check vector database
python -c "from rag_pipeline.document_processor import VectorStore; \
           vs = VectorStore(); print(vs.get_collection_stats())"
```

### Issue: "Out of memory"

```python
# Use CPU instead
config["device"] = "cpu"
```

## 📝 API Reference

### `UnifiedAgent`

#### `__init__(config, llm_engine, vector_store)`

Initialize the unified agent.

#### `process_message(query, image_path=None)`

Main entry point for all interactions.

**Args:**

- `query` (str): User's question or request
- `image_path` (str, optional): Path to CXR image

**Returns:**

- Dict with answer, thinking, image_analysis, sources, metadata

#### `get_conversation_history()`

Get all previous interactions.

#### `clear_history()`

Clear conversation history and image context.

#### `get_current_image_context()`

Get cached image analysis (if any).

## 🎓 Best Practices

1. **Natural Language** - Ask as you would ask a doctor
2. **Context Aware** - Follow-up questions work naturally
3. **Be Specific** - "What diseases?" vs "Analyze this"
4. **Use Examples** - Try quick example buttons first
5. **Check Sources** - Expand to see medical references

## 🚦 Migration Guide

### From `app.py` to `app_unified.py`

**Old Code:**

```python
# Tab 1 - Load classifier manually
binary_classifier = load_binary_classifier()
result = classifier.predict(image)

# Tab 2 - Initialize RAG separately
rag_system = initialize_rag_system()
response = rag_system.process_query(query)

# Tab 3 - Combine manually
binary_result = ...
rag_result = ...
# Manual combination
```

**New Code:**

```python
# Single unified interface
agent = UnifiedAgent(config)

# Everything through one method
response = await agent.process_message(
    query="Your question",
    image_path="optional_image.jpg"
)
# LLM handles everything
```

## 📚 Additional Resources

- **Architecture Diagrams**: See `docs/ARCHITECTURE_DIAGRAMS.md`
- **MCP Server**: See `docs/MCP_SERVER_GUIDE.md`
- **Deployment**: See `docs/PRODUCTION_DEPLOYMENT.md`

## 🤝 Contributing

The unified agent is designed to be extensible:

1. Add new vision models in `model_adapters.py`
2. Extend intent detection in `unified_agent.py`
3. Add new LLM models in `qwen_agent.py`

## ⚠️ Medical Disclaimer

This AI provides educational information only. Always consult qualified healthcare professionals for medical decisions.

---

**Made with ❤️ for better medical AI**
