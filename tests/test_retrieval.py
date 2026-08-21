"""Tests for Phase 4: RAG retrieval engine."""

import logging

import pytest

from exocortex.config import Settings
from exocortex.retrieval import QueryResponse, RAGEngine

logger = logging.getLogger(__name__)


@pytest.fixture
def settings() -> Settings:
    """Settings — tests requiring real services will check availability."""
    return Settings(deepseek_api_key="test-key")


def test_query_response_dataclass():
    """QueryResponse should hold all required fields."""
    response = QueryResponse(
        answer="Test answer",
        sources=[],
        query="test question",
        num_chunks_retrieved=0,
        model="deepseek-v4-flash",
    )
    assert response.answer == "Test answer"
    assert response.query == "test question"


def test_rag_engine_initialization(settings):
    """RAGEngine should initialize without errors."""
    # Note: this will try to init ChromaDB and other clients
    # but won't make API calls
    engine = RAGEngine(settings)
    assert engine.settings == settings


def test_rag_engine_ingest_and_index_strategy_propagation(monkeypatch, tmp_path):
    """RAGEngine.ingest_and_index should pass strategy from settings or argument to ingest_pdf."""
    from unittest.mock import MagicMock, patch

    from exocortex.ingestion import Chunk

    settings = Settings(deepseek_api_key="test-key", chunking_strategy="recursive")
    engine = RAGEngine(settings)
    engine.embedding_client = MagicMock()
    engine.embedding_client.embed_documents.return_value = [[0.1] * 1024]
    engine.vector_store = MagicMock()
    engine.vector_store.find_by_file_hash.return_value = []

    sample_chunk = Chunk(
        text="Sample chunk content",
        document_id="doc123",
        filename="sample.pdf",
        page_numbers=[1],
        chunk_index=0,
        metadata={"strategy": "recursive"},
    )

    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 sample")

    with patch("exocortex.ingestion.ingest_pdf", return_value=[sample_chunk]) as mock_ingest:
        # Default: uses self.settings.chunking_strategy
        res = engine.ingest_and_index(pdf_file)
        assert res["document_id"] == "doc123"
        mock_ingest.assert_called_once_with(
            pdf_path=pdf_file,
            chunk_size=512,
            chunk_overlap=50,
            strategy="recursive",
        )

    with patch("exocortex.ingestion.ingest_pdf", return_value=[sample_chunk]) as mock_ingest:
        # Explicit strategy override
        res = engine.ingest_and_index(pdf_file, strategy="sentence_paragraph")
        assert res["document_id"] == "doc123"
        mock_ingest.assert_called_once_with(
            pdf_path=pdf_file,
            chunk_size=512,
            chunk_overlap=50,
            strategy="sentence_paragraph",
        )


def test_rag_engine_ingest_duplicate_detection(tmp_path):
    """RAGEngine.ingest_and_index should raise DuplicateDocumentError if duplicate exists and force=False."""
    from unittest.mock import MagicMock, patch

    from exocortex.ingestion import Chunk
    from exocortex.retrieval import DuplicateDocumentError

    settings = Settings(deepseek_api_key="test-key")
    engine = RAGEngine(settings)
    engine.embedding_client = MagicMock()
    engine.vector_store = MagicMock()

    pdf_file = tmp_path / "duplicate.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 duplicate content")

    # Mock vector_store.find_by_file_hash to return existing doc
    engine.vector_store.find_by_file_hash.return_value = [
        {"document_id": "existing_doc_id", "filename": "original.pdf", "chunk_count": 5}
    ]

    with pytest.raises(DuplicateDocumentError) as exc_info:
        engine.ingest_and_index(pdf_file, force=False)

    assert "Duplicate document detected" in str(exc_info.value)
    assert exc_info.value.existing_documents[0]["filename"] == "original.pdf"
    assert exc_info.value.file_hash != ""

    # When force=True, it should proceed
    sample_chunk = Chunk(
        text="Sample chunk",
        document_id="doc123",
        filename="duplicate.pdf",
        page_numbers=[1],
        chunk_index=0,
    )
    with patch("exocortex.ingestion.ingest_pdf", return_value=[sample_chunk]):
        engine.embedding_client.embed_documents.return_value = [[0.1] * 1024]
        res = engine.ingest_and_index(pdf_file, force=True)
        assert res["document_id"] == "doc123"
        assert res["chunk_count"] == 1


# --- Full Integration Test ---
# This test requires:
# 1. Ollama running with qwen3-embedding:0.6b
# 2. DeepSeek API key configured
# 3. At least one document indexed in ChromaDB


@pytest.fixture
def full_engine() -> RAGEngine:
    """Create a fully configured RAG engine."""
    settings = Settings()  # Load from .env

    if not settings.deepseek_api_key or settings.deepseek_api_key == "test-key":
        pytest.skip("No DeepSeek API key configured")

    engine = RAGEngine(settings)

    if not engine.embedding_client.health_check():
        pytest.skip("Ollama not available")

    if engine.vector_store.count() == 0:
        pytest.skip("No documents indexed — run ingestion first")

    return engine


def test_rag_query_integration(full_engine):
    """Full RAG pipeline: query → embed → search → LLM → answer."""
    response = full_engine.query("What is this book about?")

    assert isinstance(response, QueryResponse)
    assert len(response.answer) > 0
    assert response.num_chunks_retrieved > 0
    assert len(response.sources) > 0

    logger.info(f"Query: {response.query}")
    logger.info(f"Chunks retrieved: {response.num_chunks_retrieved}")
    logger.info(f"Answer: {response.answer[:300]}")
    logger.info(f"Sources: {response.sources}")
