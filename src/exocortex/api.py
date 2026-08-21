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

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from exocortex.config import get_settings
from exocortex.retrieval import DuplicateDocumentError, RAGEngine

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
    file_hash: str = ""


class DuplicateResponse(BaseModel):
    """Response body when a duplicate document is detected."""

    detail: str
    duplicate: bool
    file_hash: str
    existing_documents: list[DocumentInfo]


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


@app.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        409: {
            "model": DuplicateResponse,
            "description": "Duplicate document detected",
        }
    },
)
async def ingest_pdf(
    file: UploadFile = File(...),
    strategy: str | None = Query(
        None,
        description="Chunking strategy override (fixed, recursive, sentence_paragraph, semantic)",
    ),
    force: bool = Query(
        False,
        description="Force ingestion and indexing even if content is duplicate",
    ),
):
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
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Ingest and index (raises DuplicateDocumentError if duplicate and force=False)
        result = engine.ingest_and_index(tmp_path, strategy=strategy, force=force)

        # Save to ebooks directory for reference
        ebooks_dir = engine.settings.ebooks_path
        ebooks_dir.mkdir(parents=True, exist_ok=True)
        permanent_path = ebooks_dir / file.filename
        permanent_path.write_bytes(content)

        return IngestResponse(
            filename=file.filename,
            document_id=result["document_id"],
            chunk_count=result["chunk_count"],
            message=f"Successfully ingested '{file.filename}' into {result['chunk_count']} chunks",
        )

    except DuplicateDocumentError as e:
        return JSONResponse(
            status_code=409,
            content={
                "detail": str(e),
                "duplicate": True,
                "file_hash": e.file_hash,
                "existing_documents": e.existing_documents,
            },
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
        if tmp_path is not None and tmp_path.exists():
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
