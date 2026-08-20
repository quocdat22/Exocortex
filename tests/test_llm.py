"""Tests for Phase 4: LLM client."""

import logging

import pytest

from exocortex.config import Settings
from exocortex.llm import LLMClient, LLMResponse, _format_context, SYSTEM_PROMPT
from exocortex.vectorstore import SearchResult

logger = logging.getLogger(__name__)


@pytest.fixture
def settings() -> Settings:
    return Settings(deepseek_api_key="test-key")


@pytest.fixture
def sample_results() -> list[SearchResult]:
    return [
        SearchResult(
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"filename": "ml_book.pdf", "page_numbers": "1,2"},
            distance=0.1,
            chunk_id="doc1_0",
        ),
        SearchResult(
            text="Deep learning uses neural networks.",
            metadata={"filename": "ml_book.pdf", "page_numbers": "5"},
            distance=0.2,
            chunk_id="doc1_1",
        ),
    ]


def test_format_context(sample_results):
    """Context should include source attribution."""
    context = _format_context(sample_results)

    assert "Source 1: ml_book.pdf" in context
    assert "Source 2: ml_book.pdf" in context
    assert "Machine learning" in context
    assert "Deep learning" in context


def test_format_context_empty():
    """Empty results should produce a 'no documents' message."""
    context = _format_context([])
    assert "No relevant documents found" in context


def test_system_prompt_has_placeholder():
    """System prompt should have {context} placeholder."""
    assert "{context}" in SYSTEM_PROMPT


def test_llm_response_dataclass():
    """LLMResponse should hold all fields."""
    response = LLMResponse(
        answer="Test answer",
        sources=[{"filename": "test.pdf", "page_numbers": "1"}],
        model="deepseek-v4-flash",
        usage={"total_tokens": 100},
    )
    assert response.answer == "Test answer"
    assert len(response.sources) == 1


# --- Integration tests (require DeepSeek API key) ---


@pytest.fixture
def llm_client() -> LLMClient:
    """Create LLM client — skips if no real API key."""
    real_settings = Settings()  # Load from .env instead of the test-key fixture
    if (
        not real_settings.deepseek_api_key
        or real_settings.deepseek_api_key == "test-key"
    ):
        pytest.skip("No DeepSeek API key configured")
    return LLMClient(real_settings)


def test_llm_generate_integration(llm_client, sample_results):
    """Should generate an answer based on context."""
    response = llm_client.generate(
        query="What is machine learning?",
        search_results=sample_results,
    )

    assert isinstance(response, LLMResponse)
    assert len(response.answer) > 0
    assert len(response.sources) == 2
    logger.info(f"LLM answer: {response.answer[:200]}")
    logger.info(f"Token usage: {response.usage}")


def test_llm_health_check_integration(llm_client):
    """Health check should return True with valid API key."""
    result = llm_client.health_check()
    assert result is True
    logger.info("LLM health check: OK")
