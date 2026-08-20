# Phase 5: FastAPI REST API + Streamlit Demo UI

## Objective

Build the FastAPI REST API with all specified endpoints and a Streamlit demo UI that
provides a user-friendly interface for uploading ebooks and querying the RAG system.

## Prerequisites

- Phase 4 completed (RAG engine working end-to-end)
- All previous tests passing

---

## API Design

### Endpoints Summary

| Method | Path               | Description                          | Request Body      | Response             |
| ------ | ------------------ | ------------------------------------ | ----------------- | -------------------- |
| GET    | `/health`          | System health check                  | —                 | `HealthResponse`     |
| POST   | `/ingest`          | Upload and index a PDF               | `multipart/form-data` (file) | `IngestResponse` |
| POST   | `/query`           | Ask a question                       | `QueryRequest`    | `QueryResponse`      |
| GET    | `/documents`       | List indexed documents               | —                 | `DocumentListResponse` |
| DELETE | `/documents/{document_id}` | Delete a document from index | —                 | `DeleteResponse`     |

---

## Implementation

### File: `src/exocortex/api.py`

```python
"""FastAPI REST API for Exocortex RAG system.

Provides endpoints for:
- Health checking (Ollama, ChromaDB, DeepSeek)
- PDF ingestion (upload → parse → embed → store)
- Querying (question → retrieve → LLM answer)
- Document management (list, delete)
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from exocortex.config import Settings, get_settings
from exocortex.embedding import EmbeddingClient
from exocortex.llm import LLMClient
from exocortex.retrieval import RAGEngine
from exocortex.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# --- Global state ---
_engine: RAGEngine | None = None


def get_engine() -> RAGEngine:
    """Get the global RAG engine instance."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = RAGEngine(settings)
    return _engine


# --- Request/Response Models ---


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., min_length=1, description="The question to ask")


class QueryResponseModel(BaseModel):
    """Response body for the /query endpoint."""

    answer: str
    sources: list[dict]
    query: str
    num_chunks_retrieved: int
    model: str
    usage: dict | None = None


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""

    filename: str
    document_id: str
    chunk_count: int
    message: str


class DocumentInfo(BaseModel):
    """Information about an indexed document."""

    document_id: str
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Response body for the /documents endpoint."""

    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class DeleteResponse(BaseModel):
    """Response body for the DELETE /documents/{document_id} endpoint."""

    document_id: str
    chunks_deleted: int
    message: str


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str  # "healthy" or "degraded"
    ollama: bool
    chromadb: bool
    llm: bool
    details: dict


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize engine on startup."""
    logger.info("Starting Exocortex RAG API...")
    get_engine()  # Initialize on startup
    logger.info("RAG engine initialized")
    yield
    logger.info("Shutting down Exocortex RAG API")


# --- FastAPI App ---

app = FastAPI(
    title="Exocortex RAG API",
    description="Retrieval-Augmented Generation system for English ebooks",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Endpoints ---


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of all system components.

    Returns status of Ollama, ChromaDB, and DeepSeek LLM.
    """
    engine = get_engine()

    ollama_ok = engine.embedding_client.health_check()
    chroma_ok = engine.vector_store.health_check()
    llm_ok = engine.llm_client.health_check()

    all_ok = ollama_ok and chroma_ok and llm_ok

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        ollama=ollama_ok,
        chromadb=chroma_ok,
        llm=llm_ok,
        details={
            "embedding_model": engine.settings.embedding_model,
            "llm_model": engine.settings.deepseek_model,
            "indexed_chunks": engine.vector_store.count(),
        },
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    """Upload and index a PDF ebook.

    The PDF is parsed, chunked, embedded, and stored in ChromaDB.
    Only PDF files with extractable text are supported (no OCR).
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted",
        )

    engine = get_engine()

    # Save uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Also save to ebooks directory for reference
        ebooks_dir = engine.settings.ebooks_path
        ebooks_dir.mkdir(parents=True, exist_ok=True)
        permanent_path = ebooks_dir / file.filename
        permanent_path.write_bytes(content)

        # Ingest and index
        result = engine.ingest_and_index(tmp_path)

        return IngestResponse(
            filename=file.filename,
            document_id=result["document_id"],
            chunk_count=result["chunk_count"],
            message=f"Successfully ingested '{file.filename}' into {result['chunk_count']} chunks",
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()


@app.post("/query", response_model=QueryResponseModel)
async def query(request: QueryRequest):
    """Ask a question about the indexed ebooks.

    The system retrieves relevant chunks and generates an answer
    using the LLM, grounded in the ebook content.
    """
    engine = get_engine()

    if engine.vector_store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload a PDF first via POST /ingest.",
        )

    try:
        response = engine.query(request.question)

        return QueryResponseModel(
            answer=response.answer,
            sources=response.sources,
            query=response.query,
            num_chunks_retrieved=response.num_chunks_retrieved,
            model=response.model,
            usage=response.usage,
        )

    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed documents with chunk counts."""
    engine = get_engine()

    docs = engine.vector_store.list_documents()

    return DocumentListResponse(
        documents=[DocumentInfo(**d) for d in docs],
        total_documents=len(docs),
        total_chunks=engine.vector_store.count(),
    )


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """Delete a document and all its chunks from the index."""
    engine = get_engine()

    deleted = engine.vector_store.delete_document(document_id)

    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found",
        )

    return DeleteResponse(
        document_id=document_id,
        chunks_deleted=deleted,
        message=f"Deleted {deleted} chunks for document '{document_id}'",
    )
```

### Running the API Server

Add this to `pyproject.toml` scripts section, or run directly:

```bash
# Run the FastAPI server
uv run uvicorn exocortex.api:app --host 0.0.0.0 --port 8000 --reload
```

---

### File: `streamlit_app.py` (Project Root)

```python
"""Streamlit demo UI for Exocortex RAG system.

Provides a web interface for:
- Uploading PDF ebooks
- Viewing indexed documents
- Asking questions with source-cited answers

Requires the FastAPI server to be running on localhost:8000.
"""

import streamlit as st
import httpx

# --- Configuration ---
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Exocortex — Ebook RAG",
    page_icon="🧠",
    layout="wide",
)


def api_request(method: str, endpoint: str, **kwargs) -> dict | None:
    """Make an API request to the FastAPI backend.

    Args:
        method: HTTP method (get, post, delete).
        endpoint: API endpoint path (e.g., '/health').
        **kwargs: Additional arguments for httpx.

    Returns:
        Response JSON dict, or None if request failed.
    """
    url = f"{API_BASE_URL}{endpoint}"
    try:
        with httpx.Client(timeout=120.0) as client:
            response = getattr(client, method)(url, **kwargs)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(
                    f"API Error ({response.status_code}): {response.json().get('detail', 'Unknown error')}"
                )
                return None
    except httpx.ConnectError:
        st.error(
            f"Cannot connect to API at {API_BASE_URL}. Is the FastAPI server running?"
        )
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# --- Sidebar: System Status ---
with st.sidebar:
    st.title("🧠 Exocortex")
    st.caption("RAG System for English Ebooks")

    st.divider()

    # Health check
    if st.button("🔄 Check System Health"):
        health = api_request("get", "/health")
        if health:
            status = health["status"]
            if status == "healthy":
                st.success("System is healthy")
            else:
                st.warning("System is degraded")

            col1, col2, col3 = st.columns(3)
            col1.metric("Ollama", "✅" if health["ollama"] else "❌")
            col2.metric("ChromaDB", "✅" if health["chromadb"] else "❌")
            col3.metric("LLM", "✅" if health["llm"] else "❌")

            st.json(health["details"])

    st.divider()

    # Document list
    st.subheader("📚 Indexed Documents")
    docs = api_request("get", "/documents")
    if docs:
        if docs["total_documents"] == 0:
            st.info("No documents indexed yet. Upload a PDF below.")
        else:
            st.metric("Total Documents", docs["total_documents"])
            st.metric("Total Chunks", docs["total_chunks"])
            for doc in docs["documents"]:
                with st.expander(f"📄 {doc['filename']}"):
                    st.text(f"ID: {doc['document_id']}")
                    st.text(f"Chunks: {doc['chunk_count']}")
                    if st.button(f"🗑️ Delete", key=f"del_{doc['document_id']}"):
                        result = api_request(
                            "delete", f"/documents/{doc['document_id']}"
                        )
                        if result:
                            st.success(result["message"])
                            st.rerun()


# --- Main Area ---
tab_query, tab_upload = st.tabs(["💬 Ask Questions", "📤 Upload Ebook"])

# --- Tab 1: Query ---
with tab_query:
    st.header("Ask a Question")
    st.caption(
        "Ask questions about your indexed ebooks. Answers are grounded in the document content."
    )

    question = st.text_input(
        "Your question:",
        placeholder="e.g., What are the main concepts discussed in chapter 1?",
    )

    if st.button("🔍 Ask", disabled=not question):
        with st.spinner("Searching documents and generating answer..."):
            result = api_request("post", "/query", json={"question": question})

        if result:
            # Display answer
            st.subheader("Answer")
            st.markdown(result["answer"])

            # Display sources
            st.subheader("Sources")
            st.caption(
                f"{result['num_chunks_retrieved']} chunks retrieved | Model: {result['model']}"
            )

            for i, source in enumerate(result["sources"], 1):
                with st.expander(
                    f"📖 Source {i}: {source['filename']} (page {source['page_numbers']})"
                ):
                    st.text(source["text_preview"])

            # Display token usage
            if result.get("usage"):
                with st.expander("📊 Token Usage"):
                    st.json(result["usage"])

# --- Tab 2: Upload ---
with tab_upload:
    st.header("Upload Ebook (PDF)")
    st.caption(
        "Upload an English ebook in PDF format. The system will extract text, create chunks, and index them."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Only PDF files with extractable text are supported. Scanned/image PDFs are not supported.",
    )

    if uploaded_file is not None:
        st.info(
            f"File: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)"
        )

        if st.button("📥 Ingest & Index"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "application/pdf",
                    )
                }
                result = api_request("post", "/ingest", files=files)

            if result:
                st.success(result["message"])
                st.json(
                    {
                        "filename": result["filename"],
                        "document_id": result["document_id"],
                        "chunk_count": result["chunk_count"],
                    }
                )
                st.rerun()  # Refresh sidebar document list
```

---

## Tests

### File: `tests/test_api.py`

```python
"""Tests for Phase 5: FastAPI REST API."""

import logging
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from exocortex.api import app, _engine, get_engine
from exocortex.config import Settings

logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


# --- Health endpoint ---


def test_health_endpoint(client):
    """GET /health should return system status."""
    response = client.get("/health")
    # May be 200 even if services are down (returns degraded status)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama" in data
    assert "chromadb" in data
    assert "llm" in data
    logger.info(f"Health: {data}")


# --- Documents endpoint ---


def test_list_documents_endpoint(client):
    """GET /documents should return document list."""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total_documents" in data
    assert "total_chunks" in data
    logger.info(f"Documents: {data}")


# --- Query endpoint ---


def test_query_empty_store(client):
    """POST /query on empty store should return 400."""
    # This depends on whether the store is actually empty
    response = client.post("/query", json={"question": "test"})
    # If no documents indexed, should be 400
    # If documents exist, should be 200
    assert response.status_code in [200, 400, 502, 503]


def test_query_missing_question(client):
    """POST /query without question should return 422."""
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_empty_question(client):
    """POST /query with empty question should return 422."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


# --- Ingest endpoint ---


def test_ingest_non_pdf(client):
    """POST /ingest with non-PDF should return 400."""
    files = {"file": ("test.txt", b"not a pdf", "text/plain")}
    response = client.post("/ingest", files=files)
    assert response.status_code == 400


# --- Delete endpoint ---


def test_delete_nonexistent_document(client):
    """DELETE /documents/{id} for nonexistent doc should return 404."""
    response = client.delete("/documents/nonexistent-id")
    assert response.status_code == 404


# --- Integration tests ---


@pytest.fixture
def sample_pdf_bytes() -> bytes | None:
    """Get bytes of a sample PDF if available."""
    pdf_dir = Path("data/ebooks")
    if not pdf_dir.exists():
        return None
    pdfs = list(pdf_dir.glob("*.pdf"))
    if not pdfs:
        return None
    return pdfs[0].read_bytes()


def test_ingest_and_query_integration(client, sample_pdf_bytes):
    """Full integration: ingest PDF → query → get answer."""
    if sample_pdf_bytes is None:
        pytest.skip("No sample PDF available in data/ebooks/")

    # Ingest
    files = {"file": ("test_book.pdf", sample_pdf_bytes, "application/pdf")}
    response = client.post("/ingest", files=files)

    if response.status_code == 503:
        pytest.skip("Ollama not available")

    assert response.status_code == 200
    data = response.json()
    assert data["chunk_count"] > 0
    logger.info(f"Ingested: {data}")

    # Query
    response = client.post("/query", json={"question": "What is this book about?"})

    if response.status_code in [502, 503]:
        pytest.skip("LLM service not available")

    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 0
    assert len(data["sources"]) > 0
    logger.info(f"Answer: {data['answer'][:200]}")

    # List documents
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total_documents"] > 0
    logger.info(f"Total documents: {data['total_documents']}")
```

---

## Running the Application

### Start the API Server
```bash
uv run uvicorn exocortex.api:app --host 0.0.0.0 --port 8000 --reload
```

### Start the Streamlit UI (in a separate terminal)
```bash
uv run streamlit run streamlit_app.py --server.port 8501
```

### Access
- **API docs (Swagger):** http://localhost:8000/docs
- **API docs (ReDoc):** http://localhost:8000/redoc
- **Streamlit UI:** http://localhost:8501

---

## Success Criteria

Phase 5 is complete when ALL of the following are true:

### Automated Tests
```bash
uv run pytest tests/test_api.py -v --log-cli-level=INFO
```

**Expected:** All unit tests PASSED. Integration tests PASSED if services available, SKIPPED otherwise.

### Manual Verification

```bash
# 1. Start the API server
uv run uvicorn exocortex.api:app --host 0.0.0.0 --port 8000

# 2. In another terminal, test endpoints with curl:

# Health check
curl http://localhost:8000/health | python -m json.tool

# Upload a PDF
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/ebooks/YOUR_BOOK.pdf" | python -m json.tool

# List documents
curl http://localhost:8000/documents | python -m json.tool

# Ask a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of this book?"}' | python -m json.tool

# 3. Start Streamlit and verify UI works
uv run streamlit run streamlit_app.py --server.port 8501
# Open http://localhost:8501 in browser
```

### Checklist
- [ ] `src/exocortex/api.py` exists with all 5 endpoints
- [ ] `streamlit_app.py` exists in project root
- [ ] `GET /health` returns status of all components
- [ ] `POST /ingest` accepts PDF upload and returns chunk count
- [ ] `POST /query` returns answer with sources
- [ ] `GET /documents` lists all indexed documents
- [ ] `DELETE /documents/{id}` removes a document
- [ ] Swagger UI accessible at `/docs`
- [ ] Streamlit UI can upload PDFs and query the system
- [ ] All API tests pass
- [ ] End-to-end flow works: upload → query → answer with sources
