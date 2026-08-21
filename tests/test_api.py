"""Tests for Phase 5: FastAPI REST API."""

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from exocortex.api import app
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


def test_ingest_endpoint_with_strategy(client, monkeypatch):
    """POST /ingest should pass strategy query param to engine.ingest_and_index."""
    from unittest.mock import MagicMock

    import exocortex.api as api_mod

    mock_engine = MagicMock()
    mock_engine.settings = Settings(deepseek_api_key="test-key", chunking_strategy="recursive")
    mock_engine.ingest_and_index.return_value = {
        "document_id": "test_id",
        "chunk_count": 10,
        "filename": "test.pdf",
    }
    monkeypatch.setattr(api_mod, "_engine", mock_engine)

    files = {"file": ("test.pdf", b"%PDF-1.4 sample content", "application/pdf")}
    response = client.post("/ingest?strategy=sentence_paragraph", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["chunk_count"] == 10
    assert mock_engine.ingest_and_index.called
    assert mock_engine.ingest_and_index.call_args.kwargs.get("strategy") == "sentence_paragraph"


def test_ingest_duplicate_detected_returns_409(client, monkeypatch):
    """POST /ingest with duplicate file and force=False should return 409 Conflict."""
    from unittest.mock import MagicMock

    import exocortex.api as api_mod
    from exocortex.retrieval import DuplicateDocumentError

    mock_engine = MagicMock()
    mock_engine.settings = Settings(deepseek_api_key="test-key")
    mock_engine.ingest_and_index.side_effect = DuplicateDocumentError(
        "Duplicate document detected",
        file_hash="testhash123",
        existing_documents=[{"document_id": "doc1", "filename": "existing.pdf", "chunk_count": 5}],
    )
    monkeypatch.setattr(api_mod, "_engine", mock_engine)

    files = {"file": ("test.pdf", b"%PDF-1.4 sample content", "application/pdf")}
    response = client.post("/ingest", files=files)
    assert response.status_code == 409
    data = response.json()
    assert data["duplicate"] is True
    assert data["file_hash"] == "testhash123"
    assert len(data["existing_documents"]) == 1
    assert data["existing_documents"][0]["filename"] == "existing.pdf"


def test_ingest_duplicate_with_force_returns_200(client, monkeypatch):
    """POST /ingest with force=True should succeed and return 200."""
    from unittest.mock import MagicMock

    import exocortex.api as api_mod

    mock_engine = MagicMock()
    mock_engine.settings = Settings(deepseek_api_key="test-key")
    mock_engine.ingest_and_index.return_value = {
        "document_id": "doc_forced",
        "chunk_count": 8,
        "filename": "test.pdf",
    }
    monkeypatch.setattr(api_mod, "_engine", mock_engine)

    files = {"file": ("test.pdf", b"%PDF-1.4 sample content", "application/pdf")}
    response = client.post("/ingest?force=true", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["chunk_count"] == 8
    assert mock_engine.ingest_and_index.called
    assert mock_engine.ingest_and_index.call_args.kwargs.get("force") is True


# --- Delete endpoint ---


def test_delete_nonexistent_document(client):
    """DELETE /documents/{id} for nonexistent doc should return 404."""
    response = client.delete("/documents/nonexistent-id")
    assert response.status_code == 404


# --- Session endpoints ---


def test_create_and_list_sessions_endpoints(client):
    """POST /sessions and GET /sessions should manage chat sessions."""
    # Create
    resp = client.post("/sessions", json={"title": "Test API Chat"})
    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["title"] == "Test API Chat"
    session_id = data["id"]

    # List
    list_resp = client.get("/sessions")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total_sessions"] >= 1
    assert any(s["id"] == session_id for s in list_data["sessions"])

    # Get details
    get_resp = client.get(f"/sessions/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == session_id
    assert get_resp.json()["messages"] == []

    # Delete
    del_resp = client.delete(f"/sessions/{session_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["session_id"] == session_id

    # Verify 404 after deletion
    assert client.get(f"/sessions/{session_id}").status_code == 404


def test_get_nonexistent_session_returns_404(client):
    """GET /sessions/{id} for nonexistent session should return 404."""
    resp = client.get("/sessions/nonexistent-session-id")
    assert resp.status_code == 404


def test_delete_nonexistent_session_returns_404(client):
    """DELETE /sessions/{id} for nonexistent session should return 404."""
    resp = client.delete("/sessions/nonexistent-session-id")
    assert resp.status_code == 404


def test_session_chat_endpoint(client, monkeypatch):
    """POST /sessions/{id}/chat should execute chat turn and return ChatResponseModel."""
    from unittest.mock import MagicMock
    import exocortex.api as api_mod
    from exocortex.retrieval import ChatResponse

    mock_engine = MagicMock()
    mock_engine.chat.return_value = ChatResponse(
        answer="Multi-turn answer from mock",
        sources=[{"filename": "test.pdf", "page_numbers": "1", "text_preview": "..."}],
        query="What is Chapter 1 about?",
        standalone_query="What is Chapter 1 about?",
        needs_retrieval=True,
        session_id="mock-session-123",
        num_chunks_retrieved=1,
        model="deepseek-v4-flash",
        usage={"total_tokens": 80},
    )
    monkeypatch.setattr(api_mod, "_engine", mock_engine)

    resp = client.post(
        "/sessions/mock-session-123/chat",
        json={"question": "What is Chapter 1 about?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Multi-turn answer from mock"
    assert data["standalone_query"] == "What is Chapter 1 about?"
    assert data["session_id"] == "mock-session-123"
    assert data["needs_retrieval"] is True
    assert mock_engine.chat.called


def test_session_chat_error_handling(client, monkeypatch):
    """POST /sessions/{id}/chat handles ConnectionError, RuntimeError, Exception."""
    from unittest.mock import MagicMock
    import exocortex.api as api_mod

    # ConnectionError -> 503
    mock_engine = MagicMock()
    mock_engine.chat.side_effect = ConnectionError("Service unreachable")
    monkeypatch.setattr(api_mod, "_engine", mock_engine)
    resp = client.post("/sessions/s1/chat", json={"question": "hello"})
    assert resp.status_code == 503

    # RuntimeError -> 502
    mock_engine.chat.side_effect = RuntimeError("Bad gateway / inference failed")
    resp = client.post("/sessions/s1/chat", json={"question": "hello"})
    assert resp.status_code == 502

    # Generic Exception -> 500
    mock_engine.chat.side_effect = ValueError("Unexpected internal state")
    resp = client.post("/sessions/s1/chat", json={"question": "hello"})
    assert resp.status_code == 500


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

    # Ingest with force=true to guarantee ingestion even if file is already in store
    files = {"file": ("test_book.pdf", sample_pdf_bytes, "application/pdf")}
    response = client.post("/ingest?force=true", files=files)

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

