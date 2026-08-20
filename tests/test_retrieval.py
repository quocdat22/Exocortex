"""Tests for Phase 4: RAG retrieval engine."""

import logging

import pytest

from exocortex.config import Settings
from exocortex.retrieval import RAGEngine, QueryResponse

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
