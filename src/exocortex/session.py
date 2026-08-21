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
from datetime import UTC, datetime
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
    return datetime.now(UTC).isoformat()


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
