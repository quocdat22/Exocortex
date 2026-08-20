# Exocortex

RAG system for English ebooks. Upload PDF ebooks (with text layer), index their
content, and ask questions answered by an LLM grounded in the ebook content.

See `docs/00-overview.md` for the full project overview and development phases.

## Setup

```bash
uv sync                       # install dependencies
cp .env.example .env          # configure your DeepSeek API key
```

## Running

Ensure Ollama is running with `qwen3-embedding:0.6b` pulled, and your DeepSeek
API key is set in `.env`.

Start the FastAPI server:

```bash
uv run uvicorn exocortex.api:app --host 0.0.0.0 --port 8000 --reload
```

In a separate terminal, start the Streamlit UI:

```bash
uv run streamlit run streamlit_app.py --server.port 8501
```

### Access

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Streamlit UI:** http://localhost:8501

## Tests

```bash
uv run pytest tests/test_api.py -v --log-cli-level=INFO
```
