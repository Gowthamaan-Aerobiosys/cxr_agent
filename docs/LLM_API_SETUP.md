# LLM Engine API Configuration Guide

### 1. OpenAI (GPT Models)

- **Default Model**: `gpt-4-turbo-preview`
- **Other Models**: `gpt-4`, `gpt-4o`, `gpt-3.5-turbo`
- **Get API Key**: https://platform.openai.com/api-keys

### 2. Anthropic (Claude Models)

- **Default Model**: `claude-3-5-sonnet-20241022`
- **Other Models**: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- **Get API Key**: https://console.anthropic.com/

### 3. Google (Gemini Models)

- **Default Model**: `gemini-1.5-pro`
- **Other Models**: `gemini-1.5-flash`, `gemini-1.0-pro`
- **Get API Key**: https://makersuite.google.com/app/apikey

## Setup Instructions

### Step 1: Install Required Packages

```powershell
# Activate your virtual environment first
.\cenv\Scripts\activate

# Install the API client libraries
pip install openai anthropic google-generativeai
```

### Step 2: Configure API Keys

1. Copy the example environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Edit `.env` and add your API keys:

   ```env
   # Choose ONE provider and set its API key

   # Option 1: OpenAI
   OPENAI_API_KEY=sk-your-actual-openai-key-here
   LLM_PROVIDER=openai

   # Option 2: Anthropic
   ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key-here
   LLM_PROVIDER=anthropic

   # Option 3: Google Gemini
   GOOGLE_API_KEY=your-actual-google-key-here
   LLM_PROVIDER=gemini
   ```

3. (Optional) Customize generation parameters:
   ```env
   LLM_MODEL_NAME=gpt-4  # Override default model
   LLM_MAX_TOKENS=2048   # Maximum response length
   LLM_TEMPERATURE=0.7   # Creativity (0.0 = deterministic, 1.0 = creative)
   ```

### Step 3: Update Your Code

The new API-based LLMEngine is initialized as follows:

```python
from src.rag.llm_engine import LLMEngine
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM Engine
llm_engine = LLMEngine(
    provider=os.getenv("LLM_PROVIDER", "openai"),  # 'openai', 'anthropic', or 'gemini'
    model_name=os.getenv("LLM_MODEL_NAME", None),  # Optional: override default model
    api_key=None,  # Optional: will read from environment if not provided
    max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
)
```

## Usage Examples

### Basic RAG Query

```python
from src.rag.llm_engine import LLMEngine, AgenticRAG
from src.rag.document_processor import VectorStore

# Initialize components
vector_store = VectorStore(collection_name="respiratory_care_docs")
llm_engine = LLMEngine(provider="openai")  # Uses OPENAI_API_KEY from .env

# Create RAG pipeline
rag = AgenticRAG(vector_store=vector_store, llm_agent=llm_engine)

# Process a query
response = rag.process_query("What are the indications for mechanical ventilation?")

print(f"Answer: {response['answer']}")
print(f"Sources: {len(response['sources'])} references")
```

### Switching Between Providers

You can easily switch between providers by changing the environment variable:

```python
# Use OpenAI
llm_engine = LLMEngine(provider="openai", model_name="gpt-4")

# Use Anthropic Claude
llm_engine = LLMEngine(provider="anthropic", model_name="claude-3-5-sonnet-20241022")

# Use Google Gemini
llm_engine = LLMEngine(provider="gemini", model_name="gemini-1.5-pro")
```

## Migration from Local Models

### Old Code (Local Model)

```python
# Old way - required GPU and local model download
llm_engine = LLMEngine(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    load_in_4bit=True,
    use_tgi=True
)
```

### New Code (API-based)

```python
# New way - uses cloud APIs
llm_engine = LLMEngine(
    provider="openai",
    model_name="gpt-4-turbo-preview"
)
```

## Cost Considerations

API usage incurs costs. Here are approximate costs per 1M tokens:

| Provider  | Model             | Input Cost | Output Cost |
| --------- | ----------------- | ---------- | ----------- |
| OpenAI    | GPT-4 Turbo       | $10        | $30         |
| OpenAI    | GPT-3.5 Turbo     | $0.50      | $1.50       |
| Anthropic | Claude 3.5 Sonnet | $3         | $15         |
| Anthropic | Claude 3 Haiku    | $0.25      | $1.25       |
| Google    | Gemini 1.5 Pro    | $3.50      | $10.50      |
| Google    | Gemini 1.5 Flash  | $0.35      | $1.05       |

**Tip**: Start with GPT-3.5 Turbo, Claude Haiku, or Gemini Flash for testing, then upgrade to more powerful models for production.

## Troubleshooting

### API Key Not Found

**Error**: `ValueError: OpenAI API key not found`

**Solution**:

1. Ensure your `.env` file exists and contains the API key
2. Check that you're loading environment variables with `load_dotenv()`
3. Verify the API key is valid and not expired

### Rate Limits

**Error**: API rate limit exceeded

**Solution**:

1. Check your API usage quota on the provider's dashboard
2. Implement request throttling in production
3. Upgrade your API plan if needed

### Import Errors

**Error**: `ImportError: anthropic package not installed`

**Solution**:

```powershell
pip install anthropic
# or for all providers:
pip install openai anthropic google-generativeai
```

## Advanced Configuration

### Custom System Prompts

You can customize the system prompt for your specific use case:

```python
class CustomLLMEngine(LLMEngine):
    def format_system_prompt(self) -> str:
        return """Your custom system prompt here..."""
```

### Streaming Responses

For long responses, you can implement streaming (provider-dependent):

```python
# Example with OpenAI streaming
response = self.client.chat.completions.create(
    model=self.model_name,
    messages=[...],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end='')
```

## Support

For issues or questions:

1. Check the [main README](../README.md)
2. Review provider-specific documentation
3. Open an issue on GitHub

## Security Best Practices

⚠️ **Never commit API keys to version control**

1. Always use `.env` files (which are in `.gitignore`)
2. Rotate API keys regularly
3. Use separate keys for development and production
4. Monitor API usage for anomalies
5. Set spending limits on provider dashboards
