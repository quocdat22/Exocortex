# Conversation History & Session Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-turn conversation history and SQLite session management with LLM query rewriting and router to Exocortex RAG.

**Architecture:** SQLite-backed SessionStore for sessions & messages persistence; LLM-based query rewriting and light routing to generate standalone queries from recent history (sliding window K=3-5 pairs); RAGEngine.chat multi-turn pipeline; RESTful FastAPI `/sessions` endpoints; and ChatGPT-style Streamlit chat UI.

**Tech Stack:** Python 3.12, SQLite (standard library `sqlite3`), FastAPI, Pydantic, OpenAI SDK (DeepSeek), Ollama, ChromaDB, Streamlit, Pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-conversation-history-design.md`

## Global Constraints

- Use Python 3.12 syntax and type hinting (`from __future__ import annotations`, `list[T]`, `T | None`).
- Use Python's built-in `sqlite3` for session storage — zero external database daemon requirements.
- Maintain 100% backward compatibility for `POST /query` and existing single-turn test suites.
- Follow TDD (Test-Driven Development): write failing unit test, verify failure, implement minimal code, verify pass, commit.

---

### Task 1: Configuration & Settings Updates

**Files:**
- Modify: `src/exocortex/config.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.sessions_db_path: str = "./data/sessions.db"`, `Settings.chat_history_window: int = 3`, `Settings.sessions_path: Path`.

- [ ] **Step 1: Write the failing test in `tests/test_config.py`**

```python
# Add to tests/test_config.py
def test_settings_session_defaults():
    """Settings should include default session database path and history window."""
    settings = Settings(deepseek_api_key="test-key")
    assert settings.sessions_db_path == "./data/sessions.db"
    assert settings.chat_history_window == 3
    assert settings.sessions_path.name == "sessions.db"


def test_settings_session_env_override(monkeypatch):
    """Settings should allow overriding session settings via environment variables."""
    monkeypatch.setenv("SESSIONS_DB_PATH", "./custom/sessions.db")
    monkeypatch.setenv("CHAT_HISTORY_WINDOW", "5")

    settings = Settings(deepseek_api_key="test-key")
    assert settings.sessions_db_path == "./custom/sessions.db"
    assert settings.chat_history_window == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_settings_session_defaults -v`  
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'sessions_db_path'`

- [ ] **Step 3: Modify `src/exocortex/config.py`**

Add session configuration fields and `sessions_path` property to `Settings` in `src/exocortex/config.py`:

```python
    # --- Session & History ---
    sessions_db_path: str = "./data/sessions.db"
    chat_history_window: int = 3  # number of recent Q&A turns (pairs) to provide

    @property
    def sessions_path(self) -> Path:
        """Return sessions SQLite database path as a Path object."""
        return Path(self.sessions_db_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`  
Expected: All tests in `tests/test_config.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add src/exocortex/config.py tests/test_config.py
git commit -m "feat(config): add session database and history window configuration"
```

---

### Task 2: SQLite Session & Message Storage (`SessionStore`)

**Files:**
- Create: `src/exocortex/session.py`
- Create: `tests/test_session.py`

**Interfaces:**
- Produces:
  - `@dataclass class Message`: `id: str`, `session_id: str`, `role: str`, `content: str`, `standalone_query: str | None`, `needs_retrieval: bool`, `sources: list[dict]`, `model: str | None`, `usage: dict | None`, `created_at: str`
  - `@dataclass class Session`: `id: str`, `title: str`, `created_at: str`, `updated_at: str`, `messages: list[Message]`
  - `class SessionStore`:
    - `__init__(db_path: str | Path = "./data/sessions.db")`
    - `create_session(title: str | None = None) -> Session`
    - `get_session(session_id: str, include_messages: bool = True) -> Session | None`
    - `list_sessions() -> list[Session]`
    - `update_session_title(session_id: str, title: str) -> bool`
    - `delete_session(session_id: str) -> bool`
    - `add_message(session_id: str, role: str, content: str, standalone_query: str | None = None, needs_retrieval: bool = True, sources: list[dict] | None = None, model: str | None = None, usage: dict | None = None) -> Message`
    - `get_recent_messages(session_id: str, limit: int = 6) -> list[Message]`

- [ ] **Step 1: Write unit tests in `tests/test_session.py`**

```python
"""Tests for Session & Message SQLite storage."""

from __future__ import annotations

import pytest
from exocortex.session import Message, Session, SessionStore


@pytest.fixture
def store(tmp_path) -> SessionStore:
    """Create a SessionStore with a temporary SQLite database."""
    db_file = tmp_path / "test_sessions.db"
    return SessionStore(db_file)


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

    msg2 = store.add_message(
        session_id=session.id,
        role="assistant",
        content="Supervised learning uses labeled data.",
        standalone_query="What is supervised learning?",
        needs_retrieval=True,
        sources=[{"filename": "ml.pdf", "page_numbers": "1"}],
        model="deepseek-v4-flash",
        usage={"total_tokens": 120},
    )
    assert msg2.role == "assistant"
    assert msg2.sources == [{"filename": "ml.pdf", "page_numbers": "1"}]
    assert msg2.model == "deepseek-v4-flash"
    assert msg2.usage == {"total_tokens": 120}

    session_with_msgs = store.get_session(session.id, include_messages=True)
    assert session_with_msgs is not None
    assert len(session_with_msgs.messages) == 2
    assert session_with_msgs.messages[0].content == "What is supervised learning?"
    assert session_with_msgs.messages[1].content == "Supervised learning uses labeled data."


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_session.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'exocortex.session'`

- [ ] **Step 3: Implement `src/exocortex/session.py`**

```python
"""SQLite-backed session and conversation message storage.

Provides durable persistence for multi-turn chat sessions, message history,
source citations, and token usage statistics.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single conversation turn (user, assistant, or system)."""

    id: str
    session_id: str
    role: str  # "user" | "assistant" | "system"
    content: str
    standalone_query: str | None = None
    needs_retrieval: bool = True
    sources: list[dict] = field(default_factory=list)
    model: str | None = None
    usage: dict | None = None
    created_at: str = ""


@dataclass
class Session:
    """A chat session containing metadata and message history."""

    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[Message] = field(default_factory=list)


def _iso_now() -> str:
    """Get current UTC timestamp as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


class SessionStore:
    """Thread-safe SQLite store for chat sessions and message logs."""

    def __init__(self, db_path: str | Path = "./data/sessions.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a configured SQLite database connection."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite database tables and indices if not already present."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    standalone_query TEXT,
                    needs_retrieval INTEGER DEFAULT 1,
                    sources_json TEXT,
                    model TEXT,
                    usage_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at);
                """
            )

    def create_session(self, title: str | None = None) -> Session:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())
        now = _iso_now()
        session_title = title or "New Conversation"

        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, session_title, now, now),
            )

        return Session(
            id=session_id,
            title=session_title,
            created_at=now,
            updated_at=now,
            messages=[],
        )

    def get_session(self, session_id: str, include_messages: bool = True) -> Session | None:
        """Retrieve a session by ID with optional messages."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()

            if not row:
                return None

            messages: list[Message] = []
            if include_messages:
                msg_rows = conn.execute(
                    """
                    SELECT id, session_id, role, content, standalone_query,
                           needs_retrieval, sources_json, model, usage_json, created_at
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                    """,
                    (session_id,),
                ).fetchall()

                for m in msg_rows:
                    sources = json.loads(m["sources_json"]) if m["sources_json"] else []
                    usage = json.loads(m["usage_json"]) if m["usage_json"] else None
                    messages.append(
                        Message(
                            id=m["id"],
                            session_id=m["session_id"],
                            role=m["role"],
                            content=m["content"],
                            standalone_query=m["standalone_query"],
                            needs_retrieval=bool(m["needs_retrieval"]),
                            sources=sources,
                            model=m["model"],
                            usage=usage,
                            created_at=m["created_at"],
                        )
                    )

            return Session(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                messages=messages,
            )

    def list_sessions(self) -> list[Session]:
        """List all chat sessions ordered by updated_at descending."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()

            return [
                Session(
                    id=r["id"],
                    title=r["title"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    messages=[],
                )
                for r in rows
            ]

    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update a session's title."""
        now = _iso_now()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now, session_id),
            )
            return cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages (via ON DELETE CASCADE)."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cursor.rowcount > 0

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        standalone_query: str | None = None,
        needs_retrieval: bool = True,
        sources: list[dict] | None = None,
        model: str | None = None,
        usage: dict | None = None,
    ) -> Message:
        """Add a message to a session and update the session's updated_at timestamp."""
        msg_id = str(uuid.uuid4())
        now = _iso_now()
        sources_json = json.dumps(sources) if sources is not None else None
        usage_json = json.dumps(usage) if usage is not None else None

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (
                    id, session_id, role, content, standalone_query,
                    needs_retrieval, sources_json, model, usage_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg_id,
                    session_id,
                    role,
                    content,
                    standalone_query,
                    1 if needs_retrieval else 0,
                    sources_json,
                    model,
                    usage_json,
                    now,
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )

        return Message(
            id=msg_id,
            session_id=session_id,
            role=role,
            content=content,
            standalone_query=standalone_query,
            needs_retrieval=needs_retrieval,
            sources=sources or [],
            model=model,
            usage=usage,
            created_at=now,
        )

    def get_recent_messages(self, session_id: str, limit: int = 6) -> list[Message]:
        """Fetch the last `limit` messages for a session in chronological order."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, role, content, standalone_query,
                       needs_retrieval, sources_json, model, usage_json, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

            messages = []
            for m in reversed(rows):
                sources = json.loads(m["sources_json"]) if m["sources_json"] else []
                usage = json.loads(m["usage_json"]) if m["usage_json"] else None
                messages.append(
                    Message(
                        id=m["id"],
                        session_id=m["session_id"],
                        role=m["role"],
                        content=m["content"],
                        standalone_query=m["standalone_query"],
                        needs_retrieval=bool(m["needs_retrieval"]),
                        sources=sources,
                        model=m["model"],
                        usage=usage,
                        created_at=m["created_at"],
                    )
                )
            return messages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_session.py -v`  
Expected: All tests in `tests/test_session.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add src/exocortex/session.py tests/test_session.py
git commit -m "feat(session): add SQLite-backed SessionStore and Message persistence"
```

---

### Task 3: LLM Query Rewriter, Router & History Context

**Files:**
- Modify: `src/exocortex/llm.py`
- Modify: `tests/test_llm.py`

**Interfaces:**
- Consumes: `Message` from `exocortex.session`, `SearchResult` from `exocortex.vectorstore`
- Produces:
  - `LLMClient.rewrite_and_route(history: list[Message], question: str) -> tuple[str, bool]`
  - `LLMClient.generate_with_history(messages_history: list[Message], query: str, search_results: list[SearchResult]) -> LLMResponse`

- [ ] **Step 1: Write unit tests in `tests/test_llm.py`**

Add unit tests in `tests/test_llm.py`:

```python
from unittest.mock import MagicMock
from exocortex.session import Message

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py::test_rewrite_and_route_empty_history -v`  
Expected: FAIL with `AttributeError: 'LLMClient' object has no attribute 'rewrite_and_route'`

- [ ] **Step 3: Update `src/exocortex/llm.py`**

Add `REWRITE_ROUTER_PROMPT`, `rewrite_and_route`, and `generate_with_history` methods to `LLMClient`:

```python
import json
import re
from exocortex.session import Message

REWRITE_ROUTER_PROMPT = """You are an AI assistant analyzing a conversation for a RAG retrieval system.
Given the chat history and a follow-up question from the user:
1. Determine if the question needs document retrieval from the ebook vector database (needs_retrieval = true/false).
   - Set needs_retrieval to false for greetings, conversational chit-chat, requests to clarify/summarize what was ALREADY said in the chat history.
   - Set needs_retrieval to true if the question asks for factual information, book content, definitions, or new topics.
2. Rewrite the user's follow-up question into a complete, standalone question in English that incorporates any missing context or references (pronouns like 'it', 'they', 'that method', 'the previous chapter', etc.) from the conversation history. If the question is already standalone, return it unchanged.

Respond ONLY with valid JSON in this exact structure:
{
  "needs_retrieval": true,
  "standalone_query": "Standalone reformulated question here"
}
"""
```

Implement `rewrite_and_route` and `generate_with_history` on `LLMClient`:

```python
    def rewrite_and_route(
        self,
        history: list[Message],
        question: str,
    ) -> tuple[str, bool]:
        """Analyze history and rewrite follow-up question into a standalone query.

        Args:
            history: List of recent Message objects (chronological order).
            question: The user's newest follow-up question.

        Returns:
            Tuple of (standalone_query, needs_retrieval).
        """
        if not history:
            return question, True

        history_lines: list[str] = []
        for msg in history:
            history_lines.append(f"{msg.role.upper()}: {msg.content}")
        history_text = "\n".join(history_lines)

        user_content = f"Chat History:\n{history_text}\n\nNew Follow-up Question:\n{question}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REWRITE_ROUTER_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw_content = response.choices[0].message.content or "{}"
            # Clean markdown code block wraps if present
            cleaned = re.sub(r"^```json\s*", "", raw_content.strip(), flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)

            standalone = data.get("standalone_query", question).strip() or question
            needs_retrieval = bool(data.get("needs_retrieval", True))
            return standalone, needs_retrieval
        except Exception as e:
            logger.warning(f"Query rewrite and router failed ({e}), falling back to raw question: {question}")
            return question, True

    def generate_with_history(
        self,
        messages_history: list[Message],
        query: str,
        search_results: list[SearchResult],
    ) -> LLMResponse:
        """Generate an answer given conversation history and retrieved context chunks.

        Args:
            messages_history: Recent conversation Message items.
            query: The user's query.
            search_results: Retrieved chunks from the vector store.

        Returns:
            LLMResponse with answer, source info, and token usage.
        """
        context = _format_context(search_results)
        system_message = SYSTEM_PROMPT.format(context=context)

        messages_payload: list[dict[str, str]] = [
            {"role": "system", "content": system_message}
        ]

        for msg in messages_history:
            if msg.role in ("user", "assistant"):
                messages_payload.append({"role": msg.role, "content": msg.content})

        messages_payload.append({"role": "user", "content": query})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_payload,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}") from e

        answer = response.choices[0].message.content or ""

        sources = []
        for result in search_results:
            meta = result.metadata or {}
            text = result.text or ""
            sources.append(
                {
                    "filename": meta.get("filename", "unknown"),
                    "page_numbers": meta.get("page_numbers", "?"),
                    "text": text,
                    "text_preview": text[:200] + "..." if len(text) > 200 else text,
                }
            )

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            answer=answer,
            sources=sources,
            model=self.model,
            usage=usage,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py -v -k "not integration"`  
Expected: All unit tests in `tests/test_llm.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add src/exocortex/llm.py tests/test_llm.py
git commit -m "feat(llm): add query rewriting, routing and multi-turn context generation"
```

---

### Task 4: Multi-turn RAG Pipeline Orchestration (`RAGEngine.chat`)

**Files:**
- Modify: `src/exocortex/retrieval.py`
- Modify: `tests/test_retrieval.py`

**Interfaces:**
- Consumes: `SessionStore`, `LLMClient`, `EmbeddingClient`, `VectorStore`
- Produces:
  - `@dataclass class ChatResponse`: `answer: str`, `sources: list[dict]`, `query: str`, `standalone_query: str`, `needs_retrieval: bool`, `session_id: str`, `num_chunks_retrieved: int`, `model: str`, `usage: dict | None`
  - `RAGEngine.chat(session_id: str, question: str) -> ChatResponse`
  - `RAGEngine.session_store: SessionStore`

- [ ] **Step 1: Write unit tests in `tests/test_retrieval.py`**

Add tests to `tests/test_retrieval.py`:

```python
from unittest.mock import MagicMock
from exocortex.llm import LLMResponse
from exocortex.retrieval import ChatResponse
from exocortex.session import SessionStore

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


def test_rag_engine_chat_pipeline(tmp_path):
    """RAGEngine.chat should orchestrate rewrite -> retrieval -> generation -> persist."""
    settings = Settings(deepseek_api_key="test-key", sessions_db_path=str(tmp_path / "sessions.db"))
    session_store = SessionStore(settings.sessions_db_path)
    session = session_store.create_session("Initial")

    engine = RAGEngine(settings=settings, session_store=session_store)
    engine.embedding_client = MagicMock()
    engine.embedding_client.embed_query.return_value = [0.1] * 1024
    engine.vector_store = MagicMock()
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_retrieval.py::test_chat_response_dataclass -v`  
Expected: FAIL with `ImportError: cannot import name 'ChatResponse' from 'exocortex.retrieval'`

- [ ] **Step 3: Modify `src/exocortex/retrieval.py`**

Add `ChatResponse` and update `RAGEngine.__init__` and `RAGEngine.chat`:

```python
from exocortex.session import Message, SessionStore

@dataclass
class ChatResponse:
    """Response to a conversational multi-turn query within a session."""

    answer: str
    sources: list[dict]
    query: str
    standalone_query: str
    needs_retrieval: bool
    session_id: str
    num_chunks_retrieved: int
    model: str
    usage: dict | None = None
```

Update `RAGEngine`:
```python
    def __init__(
        self,
        settings: Settings,
        embedding_client: EmbeddingClient | None = None,
        vector_store: VectorStore | None = None,
        llm_client: LLMClient | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client or EmbeddingClient(settings)
        self.vector_store = vector_store or VectorStore(settings)
        self.llm_client = llm_client or LLMClient(settings)
        self.session_store = session_store or SessionStore(settings.sessions_db_path)

    def chat(self, session_id: str, question: str) -> ChatResponse:
        """Process a conversational question with history, rewrite, and retrieval routing.

        Args:
            session_id: The UUID of the conversation session.
            question: The user's latest follow-up question.

        Returns:
            ChatResponse with the answer, citations, standalone query, and session info.
        """
        logger.info(f"Processing chat turn for session {session_id}: {question[:100]}...")

        # 1. Ensure session exists
        session = self.session_store.get_session(session_id, include_messages=False)
        is_new_session = False
        if session is None:
            session = self.session_store.create_session(title=question[:40].strip() or "New Conversation")
            session_id = session.id
            is_new_session = True

        # 2. Fetch recent conversation history (sliding window: 2 * window_size)
        history_limit = max(1, self.settings.chat_history_window * 2)
        recent_history = self.session_store.get_recent_messages(session_id, limit=history_limit)

        # 3. Rewrite query and determine routing
        standalone_query, needs_retrieval = self.llm_client.rewrite_and_route(
            history=recent_history,
            question=question,
        )
        logger.info(
            f"Query routed: needs_retrieval={needs_retrieval}, standalone_query='{standalone_query[:100]}'"
        )

        # 4. Perform vector retrieval if needed
        search_results = []
        if needs_retrieval and self.vector_store.count() > 0:
            query_embedding = self.embedding_client.embed_query(standalone_query)
            search_results = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=self.settings.top_k,
            )
            logger.info(f"Retrieved {len(search_results)} chunks for standalone query")

        # 5. Generate answer using LLM with context + history
        llm_response = self.llm_client.generate_with_history(
            messages_history=recent_history,
            query=question,
            search_results=search_results,
        )

        # 6. Persist user and assistant turns to database
        self.session_store.add_message(
            session_id=session_id,
            role="user",
            content=question,
        )
        self.session_store.add_message(
            session_id=session_id,
            role="assistant",
            content=llm_response.answer,
            standalone_query=standalone_query,
            needs_retrieval=needs_retrieval,
            sources=llm_response.sources,
            model=llm_response.model,
            usage=llm_response.usage,
        )

        # Auto-update session title if it was default
        if not is_new_session and (session.title == "New Conversation" or not session.title):
            clean_title = question.split("\n")[0][:40].strip()
            if clean_title:
                self.session_store.update_session_title(session_id, clean_title)

        return ChatResponse(
            answer=llm_response.answer,
            sources=llm_response.sources,
            query=question,
            standalone_query=standalone_query,
            needs_retrieval=needs_retrieval,
            session_id=session_id,
            num_chunks_retrieved=len(search_results),
            model=llm_response.model,
            usage=llm_response.usage,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieval.py -v -k "not integration"`  
Expected: All unit tests in `tests/test_retrieval.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add src/exocortex/retrieval.py tests/test_retrieval.py
git commit -m "feat(retrieval): implement RAGEngine.chat multi-turn conversational pipeline"
```

---

### Task 5: FastAPI REST Endpoints for Sessions and Chat

**Files:**
- Modify: `src/exocortex/api.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Produces:
  - `POST /sessions` -> `SessionDetailResponse`
  - `GET /sessions` -> `SessionListResponse`
  - `GET /sessions/{session_id}` -> `SessionDetailResponse`
  - `DELETE /sessions/{session_id}` -> `DeleteSessionResponse`
  - `POST /sessions/{session_id}/chat` -> `ChatResponseModel`
  - Retains `POST /query` -> `QueryResponseModel`

- [ ] **Step 1: Write endpoint tests in `tests/test_api.py`**

Add tests to `tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py::test_create_and_list_sessions_endpoints -v`  
Expected: FAIL with 404/405 Not Found

- [ ] **Step 3: Update `src/exocortex/api.py`**

Define Session schemas and endpoints in `src/exocortex/api.py`:

```python
class CreateSessionRequest(BaseModel):
    title: str | None = Field(None, description="Optional title for the session")

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Follow-up or initial question")

class MessageModel(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    standalone_query: str | None = None
    needs_retrieval: bool = True
    sources: list[dict] = []
    model: str | None = None
    usage: dict | None = None
    created_at: str

class SessionSummaryModel(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

class SessionDetailResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageModel] = []

class SessionListResponse(BaseModel):
    sessions: list[SessionSummaryModel]
    total_sessions: int

class DeleteSessionResponse(BaseModel):
    session_id: str
    message: str

class ChatResponseModel(BaseModel):
    answer: str
    sources: list[dict]
    query: str
    standalone_query: str
    needs_retrieval: bool
    session_id: str
    num_chunks_retrieved: int
    model: str
    usage: dict | None = None
```

Add the FastAPI route handlers:

```python
@app.post("/sessions", response_model=SessionDetailResponse)
async def create_session(request: CreateSessionRequest = CreateSessionRequest()):
    """Create a new chat session."""
    engine = get_engine()
    session = engine.session_store.create_session(title=request.title)
    return SessionDetailResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=[],
    )


@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    """List all chat sessions."""
    engine = get_engine()
    sessions = engine.session_store.list_sessions()
    return SessionListResponse(
        sessions=[
            SessionSummaryModel(
                id=s.id,
                title=s.title,
                created_at=s.created_at,
                updated_at=s.updated_at,
            )
            for s in sessions
        ],
        total_sessions=len(sessions),
    )


@app.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str):
    """Get details and message history of a specific session."""
    engine = get_engine()
    session = engine.session_store.get_session(session_id, include_messages=True)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    return SessionDetailResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=[
            MessageModel(
                id=m.id,
                session_id=m.session_id,
                role=m.role,
                content=m.content,
                standalone_query=m.standalone_query,
                needs_retrieval=m.needs_retrieval,
                sources=m.sources,
                model=m.model,
                usage=m.usage,
                created_at=m.created_at,
            )
            for m in session.messages
        ],
    )


@app.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(session_id: str):
    """Delete a chat session and all its messages."""
    engine = get_engine()
    ok = engine.session_store.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return DeleteSessionResponse(
        session_id=session_id,
        message=f"Session '{session_id}' deleted successfully",
    )


@app.post("/sessions/{session_id}/chat", response_model=ChatResponseModel)
async def session_chat(session_id: str, request: ChatRequest):
    """Send a question to a conversational session."""
    engine = get_engine()
    try:
        response = engine.chat(session_id=session_id, question=request.question)
        return ChatResponseModel(
            answer=response.answer,
            sources=response.sources,
            query=response.query,
            standalone_query=response.standalone_query,
            needs_retrieval=response.needs_retrieval,
            session_id=response.session_id,
            num_chunks_retrieved=response.num_chunks_retrieved,
            model=response.model,
            usage=response.usage,
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(f"Session chat failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v -k "not integration"`  
Expected: All tests in `tests/test_api.py` PASS.

- [ ] **Step 5: Commit**

```bash
git add src/exocortex/api.py tests/test_api.py
git commit -m "feat(api): add RESTful session management and multi-turn chat endpoints"
```

---

### Task 6: Streamlit Multi-turn Chat UI with Session History

**Files:**
- Modify: `streamlit_app.py`

**Interfaces:**
- Interacts with backend `/sessions`, `/sessions/{id}`, `/sessions/{id}/chat`, `/documents`, `/health`, and `/ingest`.

- [ ] **Step 1: Update `streamlit_app.py`**

Refactor `streamlit_app.py` to provide:
1. Active session state management in `st.session_state["current_session_id"]`.
2. Sidebar session list with `➕ New Chat` and 🗑️ delete buttons.
3. Tab 1: ChatGPT-style multi-turn chat area with `st.chat_message`, `st.chat_input`, and collapsible citation & rewrite details.
4. Tab 2: PDF Upload & Ingest interface.

```python
"""Streamlit demo UI for Exocortex RAG system with multi-turn conversational support."""

import httpx
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Exocortex — Ebook RAG & Conversational Assistant",
    page_icon="🧠",
    layout="wide",
)


def api_request(method: str, endpoint: str, **kwargs) -> dict | None:
    """Make an HTTP API request to FastAPI backend."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        with httpx.Client(timeout=120.0) as client:
            response = getattr(client, method)(url, **kwargs)
            if response.status_code in (200, 409):
                return response.json()
            else:
                st.error(
                    f"API Error ({response.status_code}): {response.json().get('detail', 'Unknown error')}"
                )
                return None
    except httpx.ConnectError:
        st.error(f"Cannot connect to API at {API_BASE_URL}. Is the FastAPI server running?")
        return None
    except Exception as e:
        st.error(f"Request failed: {e}")
        return None


# Initialize session state for active chat session
if "current_session_id" not in st.session_state:
    st.session_state["current_session_id"] = None

# --- Sidebar: Sessions & System Management ---
with st.sidebar:
    st.title("🧠 Exocortex")
    st.caption("Conversational RAG for English Ebooks")

    if st.button("➕ New Chat", type="primary", use_container_width=True):
        new_sess = api_request("post", "/sessions", json={})
        if new_sess:
            st.session_state["current_session_id"] = new_sess["id"]
            st.rerun()

    st.subheader("💬 Conversations")
    sessions_resp = api_request("get", "/sessions")
    if sessions_resp and sessions_resp.get("sessions"):
        for s in sessions_resp["sessions"]:
            col_title, col_del = st.columns([4, 1])
            is_active = s["id"] == st.session_state["current_session_id"]
            prefix = "👉 " if is_active else ""
            with col_title:
                if st.button(f"{prefix}{s['title'][:25]}", key=f"sess_{s['id']}", use_container_width=True):
                    st.session_state["current_session_id"] = s["id"]
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_sess_{s['id']}"):
                    api_request("delete", f"/sessions/{s['id']}")
                    if st.session_state["current_session_id"] == s["id"]:
                        st.session_state["current_session_id"] = None
                    st.rerun()
    else:
        st.caption("No conversations yet.")

    st.divider()

    with st.expander("⚙️ System Health & Status"):
        if st.button("🔄 Check Health"):
            health = api_request("get", "/health")
            if health:
                status = health["status"]
                st.success("Healthy" if status == "healthy" else "Degraded")
                col1, col2, col3 = st.columns(3)
                col1.metric("Ollama", "✅" if health["ollama"] else "❌")
                col2.metric("ChromaDB", "✅" if health["chromadb"] else "❌")
                col3.metric("LLM", "✅" if health["llm"] else "❌")
                st.json(health["details"])

    with st.expander("📚 Indexed Documents"):
        docs = api_request("get", "/documents")
        if docs:
            st.metric("Documents", docs["total_documents"])
            st.metric("Chunks", docs["total_chunks"])
            for doc in docs["documents"]:
                st.markdown(f"**{doc['filename']}** ({doc['chunk_count']} chunks)")
                if st.button("🗑️ Delete Doc", key=f"del_doc_{doc['document_id']}"):
                    api_request("delete", f"/documents/{doc['document_id']}")
                    st.rerun()


# --- Main Area ---
tab_chat, tab_upload = st.tabs(["💬 Chat", "📤 Upload Ebook"])

with tab_chat:
    # Ensure there is a valid active session
    if not st.session_state["current_session_id"]:
        if sessions_resp and sessions_resp.get("sessions"):
            st.session_state["current_session_id"] = sessions_resp["sessions"][0]["id"]
        else:
            new_sess = api_request("post", "/sessions", json={})
            if new_sess:
                st.session_state["current_session_id"] = new_sess["id"]

    current_session = None
    if st.session_state["current_session_id"]:
        current_session = api_request("get", f"/sessions/{st.session_state['current_session_id']}")

    if current_session:
        st.header(f"💬 {current_session['title']}")

        # Render message history
        for msg in current_session.get("messages", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    with st.expander("🔍 Details & Citations"):
                        st.caption(f"**Model:** {msg.get('model', 'N/A')} | **Retrieval:** {'Active' if msg.get('needs_retrieval') else 'Bypassed'}")
                        if msg.get("standalone_query"):
                            st.caption(f"**Rewritten Query:** {msg['standalone_query']}")
                        if msg.get("sources"):
                            st.markdown("**Sources:**")
                            for idx, src in enumerate(msg["sources"], 1):
                                st.markdown(f"- **Source {idx}:** `{src.get('filename')}` (p. {src.get('page_numbers')})\n> {src.get('text_preview', '')}")
                        if msg.get("usage"):
                            st.json(msg["usage"])

    # Chat Input
    user_input = st.chat_input("Ask a question about your books...")
    if user_input:
        if current_session:
            # Render user message optimistically
            with st.chat_message("user"):
                st.markdown(user_input)

            # Send to chat endpoint
            with st.spinner("Thinking & searching books..."):
                resp = api_request(
                    "post",
                    f"/sessions/{current_session['id']}/chat",
                    json={"question": user_input},
                )
            if resp:
                st.rerun()

with tab_upload:
    st.header("Upload Ebook (PDF)")
    st.caption("Upload an English ebook in PDF format. The system will extract text, create chunks, and index them.")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        st.info(f"File: {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")

        strategy = st.selectbox(
            "Chunking Strategy",
            ["recursive", "fixed", "sentence_paragraph", "semantic"],
            index=0,
        )

        dup_state = st.session_state.get(f"dup_detected_{file_key}")
        if dup_state:
            st.warning("⚠️ **Duplicate Document Detected!**")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("⚡ Force Ingest & Index", type="primary"):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    result = api_request("post", "/ingest", files=files, params={"strategy": strategy, "force": "true"})
                    if result and not result.get("duplicate"):
                        st.session_state.pop(f"dup_detected_{file_key}", None)
                        st.success(result.get("message", "Ingested successfully"))
                        st.rerun()
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.pop(f"dup_detected_{file_key}", None)
                    st.rerun()
        else:
            if st.button("📥 Ingest & Index"):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    result = api_request("post", "/ingest", files=files, params={"strategy": strategy, "force": "false"})
                if result:
                    if result.get("duplicate"):
                        st.session_state[f"dup_detected_{file_key}"] = result
                        st.rerun()
                    else:
                        st.success(result.get("message", "Ingested successfully"))
                        st.rerun()
```

- [ ] **Step 2: Commit**

```bash
git add streamlit_app.py
git commit -m "feat(ui): update Streamlit app to support multi-turn conversational chat and sessions"
```

---

### Task 7: Documentation, Roadmap Update & Full System Verification

**Files:**
- Modify: `docs/06-limitations-roadmap.md`

- [ ] **Step 1: Update `docs/06-limitations-roadmap.md`**

Mark limitation #4 as resolved in the roadmap checklist:

```markdown
### Short-term (v0.2)
- [x] Sentence-aware chunking
- [x] Content-based document deduplication (SHA-256 raw file byte hash with duplicate warning and force ingest option)
- [x] Conversation history & SQLite session management (with LLM query rewriting & routing)
- [ ] Upgrade to larger embedding model option
```

- [ ] **Step 2: Run complete unit test suite**

Run: `uv run pytest tests/ -v -k "not integration"`  
Expected: All unit tests across all test modules PASS.

- [ ] **Step 3: Commit**

```bash
git add docs/06-limitations-roadmap.md
git commit -m "docs: mark conversation history limitation as resolved in roadmap"
```
