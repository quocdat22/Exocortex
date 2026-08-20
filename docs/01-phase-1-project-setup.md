# Phase 1: Project Setup + Configuration

## Objective

Set up the project skeleton with `uv`, define all dependencies, create the centralized
configuration module using Pydantic Settings, and verify the development environment works.

## Prerequisites

- Python 3.12+ installed
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Ollama installed and running (`ollama serve`)
- `qwen3-embedding:0.6b` model pulled (`ollama pull qwen3-embedding:0.6b`)
- DeepSeek API key obtained from https://platform.deepseek.com/

---

## Step 1: Initialize Project with uv

```bash
cd Exocortex
uv init --package --name exocortex
```

This creates `pyproject.toml` and `src/exocortex/__init__.py`.

---

## Step 2: pyproject.toml

Update `pyproject.toml` with the following content. Key points:
- Python >=3.12 required
- All dependencies listed with version constraints
- Dev dependencies in a separate group
- Entry points for the FastAPI and Streamlit apps

```toml
[project]
name = "exocortex"
version = "0.1.0"
description = "RAG system for English ebooks"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "python-multipart>=0.0.9",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "pymupdf>=1.24.0",
    "chromadb>=1.0.0",
    "httpx>=0.27.0",
    "openai>=1.40.0",
    "streamlit>=1.38.0",
    "python-dotenv>=1.0.0",
]

[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "httpx>=0.27.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## Step 3: Create Directory Structure

Create the following directories and empty files:

```
mkdir -p src/exocortex tests data/ebooks chroma_data docs
touch src/exocortex/__init__.py
touch tests/__init__.py tests/conftest.py
```

---

## Step 4: .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Environment
.env
.venv/

# Data (user-specific)
data/ebooks/*.pdf
chroma_data/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

---

## Step 5: .env.example

This file documents all required environment variables. The actual `.env` file is gitignored.

```env
# === LLM Configuration ===
# DeepSeek API key (required)
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# DeepSeek API base URL (OpenAI-compatible)
DEEPSEEK_BASE_URL=https://api.deepseek.com

# === Ollama Configuration ===
# Ollama server URL (default: local)
OLLAMA_BASE_URL=http://localhost:11434

# === Paths ===
# Directory for ebook PDF files
EBOOKS_DIR=./data/ebooks

# Directory for ChromaDB persistent storage
CHROMA_PERSIST_DIR=./chroma_data
```

---

## Step 6: config.py — Centralized Configuration

**File:** `src/exocortex/config.py`

This is the single source of truth for all configuration. Uses Pydantic Settings to:
- Load secrets from `.env`
- Provide sensible defaults for all parameters
- Validate types automatically

```python
"""Centralized configuration for Exocortex RAG system.

All configurable parameters are defined here. Secrets are loaded from .env file.
Technical parameters have sensible defaults that can be overridden via environment
variables.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (DeepSeek) ---
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # --- Embedding (Ollama) ---
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dim: int = 1024
    embedding_num_ctx: int = 32768  # qwen3-embedding:0.6b supports 32k context

    # --- Chunking ---
    chunk_size: int = 512  # tokens per chunk
    chunk_overlap: int = 50  # overlap tokens between chunks

    # --- Retrieval ---
    top_k: int = 5  # number of chunks to retrieve

    # --- ChromaDB ---
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "exocortex_ebooks"

    # --- Paths ---
    ebooks_dir: str = "./data/ebooks"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def ebooks_path(self) -> Path:
        """Return ebooks directory as a Path object."""
        return Path(self.ebooks_dir)

    @property
    def chroma_path(self) -> Path:
        """Return ChromaDB directory as a Path object."""
        return Path(self.chroma_persist_dir)


def get_settings() -> Settings:
    """Factory function to create Settings instance.

    Use this instead of instantiating Settings directly to allow for
    dependency injection in tests.
    """
    return Settings()
```

### Configuration Reference Table

| Variable               | Env Var                | Default                    | Description                          |
| ---------------------- | ---------------------- | -------------------------- | ------------------------------------ |
| `deepseek_api_key`     | `DEEPSEEK_API_KEY`     | `""`                       | DeepSeek API key (required)          |
| `deepseek_base_url`    | `DEEPSEEK_BASE_URL`    | `https://api.deepseek.com` | DeepSeek API base URL                |
| `deepseek_model`       | `DEEPSEEK_MODEL`       | `deepseek-v4-flash`        | LLM model name                       |
| `llm_temperature`      | `LLM_TEMPERATURE`      | `0.1`                      | LLM temperature (low = deterministic)|
| `llm_max_tokens`       | `LLM_MAX_TOKENS`       | `1024`                     | Max tokens in LLM response           |
| `ollama_base_url`      | `OLLAMA_BASE_URL`      | `http://localhost:11434`   | Ollama server URL                    |
| `embedding_model`      | `EMBEDDING_MODEL`      | `qwen3-embedding:0.6b`    | Embedding model name                 |
| `embedding_dim`        | `EMBEDDING_DIM`        | `1024`                     | Embedding vector dimension           |
| `embedding_num_ctx`    | `EMBEDDING_NUM_CTX`    | `32768`                    | Embedding model context window       |
| `chunk_size`           | `CHUNK_SIZE`           | `512`                      | Tokens per chunk                     |
| `chunk_overlap`        | `CHUNK_OVERLAP`        | `50`                       | Overlap tokens between chunks        |
| `top_k`                | `TOP_K`                | `5`                        | Number of chunks to retrieve         |
| `chroma_persist_dir`   | `CHROMA_PERSIST_DIR`   | `./chroma_data`            | ChromaDB storage path                |
| `chroma_collection_name`| `CHROMA_COLLECTION_NAME`| `exocortex_ebooks`       | ChromaDB collection name             |
| `ebooks_dir`           | `EBOOKS_DIR`           | `./data/ebooks`            | Ebook PDF directory                  |
| `api_host`             | `API_HOST`             | `0.0.0.0`                  | FastAPI bind host                    |
| `api_port`             | `API_PORT`             | `8000`                     | FastAPI bind port                    |

---

## Step 7: Tests for Phase 1

**File:** `tests/test_config.py`

```python
"""Tests for Phase 1: Configuration module."""

from exocortex.config import Settings, get_settings


def test_settings_default_values():
    """Settings should load with sensible defaults."""
    settings = Settings(deepseek_api_key="test-key")

    assert settings.deepseek_api_key == "test-key"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.embedding_model == "qwen3-embedding:0.6b"
    assert settings.embedding_dim == 1024
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 50
    assert settings.top_k == 5
    assert settings.llm_temperature == 0.1
    assert settings.chroma_collection_name == "exocortex_ebooks"


def test_settings_env_override(monkeypatch):
    """Settings should be overridable via environment variables."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "override-key")
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("TOP_K", "10")

    settings = Settings()

    assert settings.deepseek_api_key == "override-key"
    assert settings.chunk_size == 256
    assert settings.top_k == 10


def test_settings_path_properties():
    """Path properties should return Path objects."""
    settings = Settings(deepseek_api_key="test-key")

    assert settings.ebooks_path.name == "ebooks"
    assert settings.chroma_path.name == "chroma_data"


def test_get_settings_returns_settings():
    """get_settings() should return a Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)
```

---

## Success Criteria

Phase 1 is complete when ALL of the following are true:

### Automated Tests
```bash
uv run pytest tests/test_config.py -v
```
**Expected:** All 4 tests PASSED.

### Manual Verification
```bash
# 1. uv can resolve and sync all dependencies
uv sync

# 2. The package is importable
uv run python -c "from exocortex.config import get_settings; s = get_settings(); print(f'Config loaded: embedding={s.embedding_model}, chunk_size={s.chunk_size}')"
```
**Expected output:**
```
Config loaded: embedding=qwen3-embedding:0.6b, chunk_size=512
```

### Checklist
- [x] `pyproject.toml` exists with all dependencies
- [x] `uv sync` completes without errors
- [x] `src/exocortex/config.py` exists with `Settings` class
- [x] `.env.example` exists documenting all env vars
- [x] `.gitignore` excludes `.env`, `chroma_data/`, ebook PDFs
- [x] `tests/test_config.py` — all tests pass
- [x] `from exocortex.config import get_settings` works
