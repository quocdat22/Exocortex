# Exocortex — RAG System for English Ebooks

## Project Summary

Exocortex is a Retrieval-Augmented Generation (RAG) system that allows users to upload
English ebooks (PDF with text layer), index their content, and ask questions answered by
an LLM grounded in the ebook content.

**Core Flow:** Upload PDF → Parse & Chunk → Embed & Store → Query → Retrieve → LLM Answer

---

## Tech Stack

| Component          | Technology                        | Notes                                      |
| ------------------ | --------------------------------- | ------------------------------------------ |
| Language           | Python 3.12+                      |                                            |
| Package Manager    | `uv`                              | Test, build, dependency management         |
| PDF Parsing        | PyMuPDF (fitz) 1.28.x             | Text extraction only, no OCR               |
| Embedding Model    | `qwen3-embedding:0.6b` via Ollama | Local, 1024-dim vectors, 32k context       |
| Vector Store       | ChromaDB 1.5.x                    | Embedded mode, persistent storage          |
| LLM                | `deepseek-v4-flash` via API       | OpenAI-compatible endpoint, temp=0.1       |
| API Framework      | FastAPI                           | REST API                                   |
| Demo UI            | Streamlit 1.61.x                  |                                            |
| Configuration      | Pydantic Settings + `.env`        | Centralized config                         |
| Testing            | pytest                            | Run via `uv run pytest`                    |

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Streamlit   │────▶│   FastAPI     │────▶│  Ingestion   │
│  Demo UI     │     │   REST API   │     │  Pipeline    │
└─────────────┘     └──────┬───────┘     └──────┬───────┘
                           │                     │
                           │                     ▼
                           │              ┌──────────────┐
                           │              │  PyMuPDF     │
                           │              │  PDF → Text  │
                           │              └──────┬───────┘
                           │                     │
                           │                     ▼
                           │              ┌──────────────┐
                           │              │  Chunking    │
                           │              │  Fixed-size  │
                           │              │  + Overlap   │
                           │              └──────┬───────┘
                           │                     │
                           ▼                     ▼
                    ┌──────────────┐     ┌──────────────┐
                    │  Retrieval   │◀───▶│  ChromaDB    │
                    │  Engine      │     │  Vector Store │
                    └──────┬───────┘     └──────────────┘
                           │                     ▲
                           │                     │
                           │              ┌──────────────┐
                           │              │  Ollama      │
                           │              │  Embedding   │
                           │              │  (qwen3)     │
                           │              └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  DeepSeek    │
                    │  LLM (API)  │
                    │  temp=0.1   │
                    └──────────────┘
```

---

## Project Structure

```
Exocortex/
├── docs/                              # Development documentation
│   ├── 00-overview.md                 # This file
│   ├── 01-phase-1-project-setup.md    # Phase 1: Project setup + config
│   ├── 02-phase-2-pdf-ingestion.md    # Phase 2: PDF parsing + chunking
│   ├── 03-phase-3-embedding-vectorstore.md  # Phase 3: Embedding + ChromaDB
│   ├── 04-phase-4-retrieval-llm.md    # Phase 4: Retrieval + LLM generation
│   ├── 05-phase-5-api-ui.md          # Phase 5: FastAPI + Streamlit
│   └── 06-limitations-roadmap.md      # Known limitations + future roadmap
├── src/exocortex/                     # Source code (Python package)
│   ├── __init__.py
│   ├── config.py                      # Pydantic Settings (centralized config)
│   ├── ingestion.py                   # PDF parsing + text chunking
│   ├── embedding.py                   # Ollama embedding client
│   ├── vectorstore.py                 # ChromaDB operations
│   ├── retrieval.py                   # Query → retrieve relevant chunks
│   ├── llm.py                         # DeepSeek LLM client
│   └── api.py                         # FastAPI application + endpoints
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures
│   ├── test_config.py                 # Phase 1 tests
│   ├── test_ingestion.py              # Phase 2 tests
│   ├── test_embedding.py             # Phase 3 tests
│   ├── test_vectorstore.py            # Phase 3 tests
│   ├── test_retrieval.py              # Phase 4 tests
│   ├── test_llm.py                    # Phase 4 tests
│   └── test_api.py                    # Phase 5 tests
├── streamlit_app.py                   # Streamlit demo UI (entry point)
├── data/                              # Ebook storage (gitignored)
│   └── ebooks/                        # Place PDF files here
├── chroma_data/                       # ChromaDB persistent storage (gitignored)
├── pyproject.toml                     # Project metadata + dependencies
├── .env.example                       # Example environment variables
├── .env                               # Actual environment variables (gitignored)
├── .gitignore
└── README.md
```

---

## Development Phases

| Phase | Name                      | Key Deliverables                                  | Doc                         |
| ----- | ------------------------- | ------------------------------------------------- | --------------------------- |
| 1     | Project Setup + Config    | `pyproject.toml`, `config.py`, `.env.example`     | `01-phase-1-project-setup.md` |
| 2     | PDF Ingestion             | `ingestion.py` — parse PDF, chunk text            | `02-phase-2-pdf-ingestion.md` |
| 3     | Embedding + Vector Store  | `embedding.py`, `vectorstore.py` — embed & store  | `03-phase-3-embedding-vectorstore.md` |
| 4     | Retrieval + LLM           | `retrieval.py`, `llm.py` — query & answer         | `04-phase-4-retrieval-llm.md` |
| 5     | API + UI                  | `api.py`, `streamlit_app.py` — endpoints & demo   | `05-phase-5-api-ui.md` |

Each phase document contains:
- **Objective** — what this phase achieves
- **Prerequisites** — what must be done before starting
- **Implementation specification** — file-by-file, function-by-function
- **Configuration values** — all settings with defaults
- **Success criteria** — observable, testable conditions
- **Test commands** — exact `uv run` commands to verify

---

## How to Use These Docs

These documents are designed for an **AI coding agent** (e.g., Claude Code) to implement
the project phase by phase. Each phase is self-contained and builds on the previous one.

**For the AI agent:**
1. Read `00-overview.md` first for context
2. Implement phases in order (1 → 2 → 3 → 4 → 5)
3. After each phase, run the success criteria tests before moving to the next
4. All configuration values are specified — use them as defaults
5. All function signatures are specified — implement them exactly
6. Read `06-limitations-roadmap.md` for known constraints

**For the human developer:**
1. Ensure Ollama is running with `qwen3-embedding:0.6b` pulled
2. Have a DeepSeek API key ready
3. Review each phase's success criteria to verify correctness
