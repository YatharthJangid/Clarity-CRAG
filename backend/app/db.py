import sqlite3
import uuid
from datetime import datetime, timezone

from app.config import settings


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.threads_db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


_conn = _connect()
_conn.execute(
    """
    CREATE TABLE IF NOT EXISTS threads (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """
)
_conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_thread() -> dict:
    thread_id = str(uuid.uuid4())
    now = _now()
    _conn.execute(
        "INSERT INTO threads (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (thread_id, "New Chat", now, now),
    )
    _conn.commit()
    return {"thread_id": thread_id, "title": "New Chat", "created_at": now, "updated_at": now}


def list_threads() -> list[dict]:
    rows = _conn.execute("SELECT * FROM threads ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def get_thread(thread_id: str) -> dict | None:
    row = _conn.execute("SELECT * FROM threads WHERE thread_id = ?", (thread_id,)).fetchone()
    return dict(row) if row else None


def touch_thread(thread_id: str, first_message: str | None = None) -> None:
    now = _now()
    thread = get_thread(thread_id)
    if thread and thread["title"] == "New Chat" and first_message:
        title = first_message.strip().replace("\n", " ")[:50]
        _conn.execute(
            "UPDATE threads SET updated_at = ?, title = ? WHERE thread_id = ?",
            (now, title, thread_id),
        )
    else:
        _conn.execute("UPDATE threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id))
    _conn.commit()
