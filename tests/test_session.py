"""Tests for Session & Message SQLite storage."""

from __future__ import annotations

import pytest

from exocortex.session import Message, Session, SessionStore


@pytest.fixture
def store(tmp_path) -> SessionStore:
    """Create a SessionStore with a temporary SQLite database."""
    db_file = tmp_path / "subdir" / "test_sessions.db"
    return SessionStore(db_file)


def test_message_dataclass_defaults():
    """Verify default field values on Message dataclass."""
    msg = Message(id="1", session_id="s1", role="user", content="hello")
    assert msg.standalone_query is None
    assert msg.needs_retrieval is True
    assert msg.sources == []
    assert msg.model is None
    assert msg.usage is None
    assert msg.created_at == ""


def test_session_dataclass_defaults():
    """Verify default field values on Session dataclass."""
    s = Session(id="s1", title="Test", created_at="now", updated_at="now")
    assert s.messages == []


def test_create_and_get_session(store: SessionStore):
    """Should create a new session and retrieve it."""
    session = store.create_session(title="Machine Learning Discussion")
    assert session.id is not None
    assert session.title == "Machine Learning Discussion"
    assert session.created_at is not None
    assert session.messages == []

    fetched = store.get_session(session.id)
    assert fetched is not None
    assert fetched.id == session.id
    assert fetched.title == "Machine Learning Discussion"


def test_create_session_default_title(store: SessionStore):
    """Should assign default title if none provided."""
    session = store.create_session()
    assert session.title == "New Conversation"


def test_get_nonexistent_session(store: SessionStore):
    """Should return None for nonexistent session ID."""
    assert store.get_session("nonexistent-uuid") is None


def test_get_session_without_messages(store: SessionStore):
    """Should return session with empty messages when include_messages=False."""
    session = store.create_session("Chat")
    store.add_message(session.id, "user", "Hello")
    fetched = store.get_session(session.id, include_messages=False)
    assert fetched is not None
    assert fetched.messages == []


def test_list_sessions_ordered_by_updated(store: SessionStore):
    """list_sessions should return sessions ordered by updated_at descending."""
    s1 = store.create_session("First Session")
    s2 = store.create_session("Second Session")

    sessions = store.list_sessions()
    assert len(sessions) == 2
    assert [s.id for s in sessions] == [s2.id, s1.id]


def test_update_session_title(store: SessionStore):
    """Should update session title."""
    session = store.create_session("Old Title")
    ok = store.update_session_title(session.id, "New Title")
    assert ok is True

    fetched = store.get_session(session.id)
    assert fetched is not None
    assert fetched.title == "New Title"


def test_update_nonexistent_session_title(store: SessionStore):
    """Updating nonexistent session title should return False."""
    assert store.update_session_title("nonexistent-id", "New Title") is False


def test_add_and_get_messages(store: SessionStore):
    """Should add user and assistant messages with metadata."""
    session = store.create_session("Chat 1")

    msg1 = store.add_message(
        session_id=session.id,
        role="user",
        content="What is supervised learning?",
    )
    assert msg1.id is not None
    assert msg1.session_id == session.id
    assert msg1.role == "user"
    assert msg1.content == "What is supervised learning?"
    assert msg1.needs_retrieval is True
    assert msg1.sources == []
    assert msg1.model is None
    assert msg1.usage is None

    msg2 = store.add_message(
        session_id=session.id,
        role="assistant",
        content="Supervised learning uses labeled data.",
        standalone_query="What is supervised learning?",
        needs_retrieval=False,
        sources=[{"filename": "ml.pdf", "page_numbers": "1"}],
        model="deepseek-v4-flash",
        usage={"total_tokens": 120},
    )
    assert msg2.role == "assistant"
    assert msg2.needs_retrieval is False
    assert msg2.sources == [{"filename": "ml.pdf", "page_numbers": "1"}]
    assert msg2.model == "deepseek-v4-flash"
    assert msg2.usage == {"total_tokens": 120}

    session_with_msgs = store.get_session(session.id, include_messages=True)
    assert session_with_msgs is not None
    assert len(session_with_msgs.messages) == 2
    assert session_with_msgs.messages[0].content == "What is supervised learning?"
    assert session_with_msgs.messages[0].needs_retrieval is True
    assert session_with_msgs.messages[1].content == "Supervised learning uses labeled data."
    assert session_with_msgs.messages[1].needs_retrieval is False


def test_get_recent_messages_sliding_window(store: SessionStore):
    """get_recent_messages should return the last N messages in chronological order."""
    session = store.create_session("Chat")
    for i in range(10):
        role = "user" if i % 2 == 0 else "assistant"
        store.add_message(session_id=session.id, role=role, content=f"Message {i}")

    recent = store.get_recent_messages(session.id, limit=4)
    assert len(recent) == 4
    assert [m.content for m in recent] == ["Message 6", "Message 7", "Message 8", "Message 9"]


def test_delete_session_cascades(store: SessionStore):
    """Deleting a session should remove the session and all its messages."""
    session = store.create_session("To Delete")
    store.add_message(session.id, "user", "Hello")
    store.add_message(session.id, "assistant", "Hi there")

    deleted = store.delete_session(session.id)
    assert deleted is True

    assert store.get_session(session.id) is None
    assert store.get_recent_messages(session.id) == []


def test_delete_nonexistent_session(store: SessionStore):
    """Deleting nonexistent session should return False."""
    assert store.delete_session("nonexistent-id") is False
