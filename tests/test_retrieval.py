"""Tests for Phase 4: RAG retrieval engine."""

import logging

import pytest

from unittest.mock import MagicMock

from exocortex.config import Settings
from exocortex.llm import LLMResponse
from exocortex.retrieval import ChatResponse, QueryResponse, RAGEngine
from exocortex.session import SessionStore

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


def test_rag_engine_initialization(tmp_path):
    """RAGEngine should initialize without errors."""
    settings = Settings(deepseek_api_key="test-key", sessions_db_path=str(tmp_path / "sessions.db"))
    engine = RAGEngine(settings)
    assert engine.settings == settings
    assert engine.session_store is not None


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


def test_chat_response_dataclass():
    """ChatResponse should hold all multi-turn metadata fields."""
    response = ChatResponse(
        answer="Answer text",
        sources=[],
        query="Follow up?",
        standalone_query="What about X?",
        needs_retrieval=True,
        session_id="s1",
        num_chunks_retrieved=3,
        model="deepseek-v4-flash",
    )
    assert response.session_id == "s1"
    assert response.standalone_query == "What about X?"
    assert response.needs_retrieval is True
    assert response.usage is None


def test_rag_engine_chat_pipeline(tmp_path):
    """RAGEngine.chat should orchestrate rewrite -> retrieval -> generation -> persist."""
    settings = Settings(deepseek_api_key="test-key", sessions_db_path=str(tmp_path / "sessions.db"))
    session_store = SessionStore(settings.sessions_db_path)
    session = session_store.create_session("Initial")

    engine = RAGEngine(settings=settings, session_store=session_store)
    engine.embedding_client = MagicMock()
    engine.embedding_client.embed_query.return_value = [0.1] * 1024
    engine.vector_store = MagicMock()
    engine.vector_store.count.return_value = 5
    engine.vector_store.query.return_value = []

    engine.llm_client = MagicMock()
    engine.llm_client.rewrite_and_route.return_value = ("Standalone Question", True)
    engine.llm_client.generate_with_history.return_value = LLMResponse(
        answer="Grounded chat answer",
        sources=[],
        model="deepseek-v4-flash",
        usage={"total_tokens": 50},
    )

    response = engine.chat(session_id=session.id, question="My Question")

    assert isinstance(response, ChatResponse)
    assert response.answer == "Grounded chat answer"
    assert response.standalone_query == "Standalone Question"
    assert response.session_id == session.id

    # Check persistence in SessionStore
    updated_session = session_store.get_session(session.id, include_messages=True)
    assert updated_session is not None
    assert len(updated_session.messages) == 2
    assert updated_session.messages[0].role == "user"
    assert updated_session.messages[0].content == "My Question"
    assert updated_session.messages[1].role == "assistant"
    assert updated_session.messages[1].content == "Grounded chat answer"
    # Auto-update title on first question
    assert updated_session.title == "My Question"


def test_rag_engine_chat_no_retrieval(tmp_path):
    """When router decides needs_retrieval=False, vector store query is bypassed."""
    settings = Settings(deepseek_api_key="test-key", sessions_db_path=str(tmp_path / "sessions.db"))
    session_store = SessionStore(settings.sessions_db_path)
    session = session_store.create_session("Chat")

    engine = RAGEngine(settings=settings, session_store=session_store)
    engine.embedding_client = MagicMock()
    engine.vector_store = MagicMock()
    engine.vector_store.count.return_value = 5

    engine.llm_client = MagicMock()
    engine.llm_client.rewrite_and_route.return_value = ("Thank you!", False)
    engine.llm_client.generate_with_history.return_value = LLMResponse(
        answer="You are welcome!",
        sources=[],
        model="deepseek-v4-flash",
    )

    response = engine.chat(session_id=session.id, question="Thank you!")

    assert response.needs_retrieval is False
    assert response.num_chunks_retrieved == 0
    assert not engine.embedding_client.embed_query.called
    assert not engine.vector_store.query.called


def test_rag_engine_chat_auto_creates_session_if_missing(tmp_path):
    """If session_id does not exist, chat should create a new session."""
    settings = Settings(deepseek_api_key="test-key", sessions_db_path=str(tmp_path / "sessions.db"))
    session_store = SessionStore(settings.sessions_db_path)

    engine = RAGEngine(settings=settings, session_store=session_store)
    engine.embedding_client = MagicMock()
    engine.embedding_client.embed_query.return_value = [0.1] * 1024
    engine.vector_store = MagicMock()
    engine.vector_store.count.return_value = 5
    engine.vector_store.query.return_value = []

    engine.llm_client = MagicMock()
    engine.llm_client.rewrite_and_route.return_value = ("Hello world", False)
    engine.llm_client.generate_with_history.return_value = LLMResponse(
        answer="Hi!",
        sources=[],
        model="deepseek-v4-flash",
    )

    response = engine.chat(session_id="nonexistent-id", question="Hello world")
    assert response.session_id != "nonexistent-id"
    created = session_store.get_session(response.session_id, include_messages=True)
    assert created is not None
    assert created.title == "Hello world"
    assert len(created.messages) == 2


def test_rag_engine_chat_empty_vectorstore(tmp_path):
    """When vector store is empty (count == 0), retrieval query is skipped."""
    settings = Settings(deepseek_api_key="test-key", sessions_db_path=str(tmp_path / "sessions.db"))
    session_store = SessionStore(settings.sessions_db_path)
    session = session_store.create_session()

    engine = RAGEngine(settings=settings, session_store=session_store)
    engine.embedding_client = MagicMock()
    engine.vector_store = MagicMock()
    engine.vector_store.count.return_value = 0

    engine.llm_client = MagicMock()
    engine.llm_client.rewrite_and_route.return_value = ("Some Question", True)
    engine.llm_client.generate_with_history.return_value = LLMResponse(
        answer="I don't have enough information.",
        sources=[],
        model="deepseek-v4-flash",
    )

    response = engine.chat(session_id=session.id, question="Some Question")
    assert response.num_chunks_retrieved == 0
    assert not engine.embedding_client.embed_query.called
    assert not engine.vector_store.query.called


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
