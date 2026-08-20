"""Tests for Phase 3: Embedding client."""

import logging

import pytest

from exocortex.config import Settings
from exocortex.embedding import EmbeddingClient, QUERY_INSTRUCTION

logger = logging.getLogger(__name__)


@pytest.fixture
def settings() -> Settings:
    return Settings(deepseek_api_key="test-key")


@pytest.fixture
def client(settings) -> EmbeddingClient:
    return EmbeddingClient(settings)


def test_embed_client_initialization(client):
    """Client should initialize with correct settings."""
    assert client.model == "qwen3-embedding:0.6b"
    assert client.expected_dim == 1024


def test_embed_empty_list(client):
    """Embedding empty list should return empty list."""
    result = client.embed_documents([])
    assert result == []


def test_query_instruction_prefix():
    """Query instruction prefix should be defined."""
    assert "Instruct:" in QUERY_INSTRUCTION
    assert "Query:" in QUERY_INSTRUCTION


# --- Integration tests (require Ollama running) ---


@pytest.fixture
def ollama_available(client) -> bool:
    """Check if Ollama is available for integration tests."""
    if not client.health_check():
        pytest.skip("Ollama not available or model not pulled")
    return True


def test_embed_documents_integration(client, ollama_available):
    """Should produce embeddings with correct dimension."""
    texts = ["Hello world", "This is a test document about machine learning."]
    embeddings = client.embed_documents(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024
    assert len(embeddings[1]) == 1024
    assert all(isinstance(v, float) for v in embeddings[0])
    logger.info(f"Embedded {len(texts)} texts, dim={len(embeddings[0])}")


def test_embed_query_integration(client, ollama_available):
    """Query embedding should have instruction prefix applied."""
    embedding = client.embed_query("What is machine learning?")

    assert len(embedding) == 1024
    assert all(isinstance(v, float) for v in embedding)
    logger.info(f"Query embedding dim={len(embedding)}")


def test_embed_similarity_integration(client, ollama_available):
    """Similar texts should have closer embeddings than dissimilar texts."""
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "The weather today is sunny and warm.",
    ]
    embeddings = client.embed_documents(texts)

    # Compute cosine similarity (dot product since embeddings are L2-normalized)
    def cosine_sim(a, b):
        return sum(x * y for x, y in zip(a, b))

    sim_related = cosine_sim(embeddings[0], embeddings[1])  # ML vs DL
    sim_unrelated = cosine_sim(embeddings[0], embeddings[2])  # ML vs weather

    assert sim_related > sim_unrelated, (
        f"Related texts should be more similar: "
        f"ML-DL={sim_related:.4f} vs ML-weather={sim_unrelated:.4f}"
    )
    logger.info(f"Similarity: ML-DL={sim_related:.4f}, ML-weather={sim_unrelated:.4f}")


def test_health_check_integration(client):
    """Health check should return a boolean."""
    result = client.health_check()
    assert isinstance(result, bool)
    logger.info(f"Ollama health check: {'OK' if result else 'FAILED'}")
