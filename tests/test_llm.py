"""Tests for Phase 4: LLM client."""

import logging

import pytest

from unittest.mock import MagicMock

from exocortex.config import Settings
from exocortex.llm import SYSTEM_PROMPT, LLMClient, LLMResponse, _format_context
from exocortex.session import Message
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


def test_rewrite_and_route_empty_history(settings):
    """When history is empty, rewrite_and_route should return raw question and True immediately."""
    client = LLMClient(settings)
    standalone, needs_retrieval = client.rewrite_and_route([], "What is machine learning?")
    assert standalone == "What is machine learning?"
    assert needs_retrieval is True


def test_rewrite_and_route_with_history(settings):
    """When history is present, should parse JSON from LLM response."""
    client = LLMClient(settings)
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"needs_retrieval": true, "standalone_query": "What are the limitations of Deep Learning?"}'))
    ]
    client.client.chat.completions.create = MagicMock(return_value=mock_response)

    history = [
        Message(id="1", session_id="s1", role="user", content="Tell me about Deep Learning."),
        Message(id="2", session_id="s1", role="assistant", content="Deep Learning uses neural networks."),
    ]
    standalone, needs_retrieval = client.rewrite_and_route(history, "What are its limitations?")

    assert standalone == "What are the limitations of Deep Learning?"
    assert needs_retrieval is True
    assert client.client.chat.completions.create.called


def test_rewrite_and_route_conversational_no_retrieval(settings):
    """When question is conversational, router should return needs_retrieval=False."""
    client = LLMClient(settings)
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"needs_retrieval": false, "standalone_query": "Thank you for the explanation"}'))
    ]
    client.client.chat.completions.create = MagicMock(return_value=mock_response)

    history = [
        Message(id="1", session_id="s1", role="user", content="Explain RAG."),
        Message(id="2", session_id="s1", role="assistant", content="RAG retrieves context before generation."),
    ]
    standalone, needs_retrieval = client.rewrite_and_route(history, "Thank you, that was helpful!")

    assert needs_retrieval is False


def test_rewrite_and_route_with_markdown_fence(settings):
    """Should handle JSON wrapped in markdown code blocks."""
    client = LLMClient(settings)
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='```json\n{"needs_retrieval": true, "standalone_query": "Explain transformers architecture"}\n```'))
    ]
    client.client.chat.completions.create = MagicMock(return_value=mock_response)

    history = [
        Message(id="1", session_id="s1", role="user", content="Tell me about Attention mechanism."),
    ]
    standalone, needs_retrieval = client.rewrite_and_route(history, "Explain transformers")

    assert standalone == "Explain transformers architecture"
    assert needs_retrieval is True


def test_rewrite_and_route_fallback_on_error(settings):
    """Should fallback to raw question and needs_retrieval=True on LLM exception or invalid JSON."""
    client = LLMClient(settings)
    client.client.chat.completions.create = MagicMock(side_effect=Exception("API Error"))

    history = [
        Message(id="1", session_id="s1", role="user", content="Hello"),
    ]
    standalone, needs_retrieval = client.rewrite_and_route(history, "What is AI?")

    assert standalone == "What is AI?"
    assert needs_retrieval is True


def test_generate_with_history_builds_messages(settings, sample_results):
    """generate_with_history should include system context, history turns, and current query."""
    client = LLMClient(settings)
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Machine learning models generalize from data."))
    ]
    mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20, total_tokens=70)
    client.client.chat.completions.create = MagicMock(return_value=mock_response)

    history = [
        Message(id="1", session_id="s1", role="user", content="What is AI?"),
        Message(id="2", session_id="s1", role="assistant", content="AI is artificial intelligence."),
    ]
    response = client.generate_with_history(
        messages_history=history,
        query="Tell me about ML.",
        search_results=sample_results,
    )

    assert response.answer == "Machine learning models generalize from data."
    assert len(response.sources) == 2
    assert client.client.chat.completions.create.called
    call_args = client.client.chat.completions.create.call_args.kwargs
    messages = call_args["messages"]
    assert len(messages) == 4  # System, User(History), Assistant(History), User(Current)
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is AI?"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "AI is artificial intelligence."
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "Tell me about ML."


def test_generate_with_history_error(settings, sample_results):
    """generate_with_history should raise RuntimeError on API failure."""
    client = LLMClient(settings)
    client.client.chat.completions.create = MagicMock(side_effect=Exception("Connection timed out"))

    with pytest.raises(RuntimeError, match="LLM API call failed"):
        client.generate_with_history(
            messages_history=[],
            query="Test query",
            search_results=sample_results,
        )


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
