"""Tagged-item store: notes, important moments, questions, commitments.

One sqlite database per agent at <agent>/items.db. The schema is the
v0.3 layout's `items` table: kind, body, source ('human' or 'agent'),
source_ref (e.g. Telegram message id), status, timestamps, JSON metadata.

Hashtags in Telegram messages produce items with source='human'. The
agent can produce items via the `tag_item` tool with source='agent'.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from return_architecture import paths


Kind = Literal["note", "important", "question", "commitment"]
KINDS: tuple[Kind, ...] = ("note", "important", "question", "commitment")

_HASHTAG_RE = re.compile(r"#(note|important|question|commitment)\b", re.IGNORECASE)


@dataclass
class Item:
    id: int
    kind: str
    body: str
    source: str
    source_ref: str | None
    status: str
    created_at: str
    resolved_at: str | None
    metadata: dict


def _db_path(slug: str) -> Path:
    return paths.agent_dir(slug) / "items.db"


def _connect(slug: str) -> sqlite3.Connection:
    paths.agent_dir(slug).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_db_path(slug))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            kind        TEXT NOT NULL,
            body        TEXT NOT NULL,
            source      TEXT NOT NULL,
            source_ref  TEXT,
            status      TEXT NOT NULL DEFAULT 'open',
            created_at  TEXT NOT NULL,
            resolved_at TEXT,
            metadata    TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_items_kind_status ON items(kind, status)"
    )
    conn.commit()


def parse_tags(text: str) -> list[str]:
    """Return the unique kinds tagged in the text, in order of first appearance."""
    seen: list[str] = []
    for m in _HASHTAG_RE.finditer(text):
        kind = m.group(1).lower()
        if kind not in seen:
            seen.append(kind)
    return seen


def strip_tags(text: str) -> str:
    return _HASHTAG_RE.sub("", text).strip()


def add_item(
    slug: str,
    *,
    kind: str,
    body: str,
    source: str,
    source_ref: str | None = None,
    metadata: dict | None = None,
) -> int:
    if kind not in KINDS:
        raise ValueError(f"Invalid kind '{kind}'. Must be one of {KINDS}.")
    if not body.strip():
        raise ValueError("Empty body.")
    conn = _connect(slug)
    try:
        _ensure_schema(conn)
        cur = conn.execute(
            """
            INSERT INTO items
              (kind, body, source, source_ref, status, created_at, metadata)
            VALUES (?, ?, ?, ?, 'open', ?, ?)
            """,
            (
                kind,
                body.strip(),
                source,
                source_ref,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(metadata) if metadata else None,
            ),
        )
        conn.commit()
        return cur.lastrowid or 0
    finally:
        conn.close()


def list_items(
    slug: str,
    *,
    kind: str | None = None,
    status: str | None = "open",
    limit: int = 50,
) -> list[Item]:
    conn = _connect(slug)
    try:
        _ensure_schema(conn)
        query = "SELECT * FROM items WHERE 1=1"
        params: list = []
        if kind is not None:
            query += " AND kind = ?"
            params.append(kind)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [_row_to_item(r) for r in rows]
    finally:
        conn.close()


def resolve_item(slug: str, item_id: int) -> bool:
    conn = _connect(slug)
    try:
        _ensure_schema(conn)
        cur = conn.execute(
            "UPDATE items SET status='resolved', resolved_at=? "
            "WHERE id=? AND status='open'",
            (datetime.now(timezone.utc).isoformat(), item_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def update_item(
    slug: str,
    item_id: int,
    *,
    body: str | None = None,
    metadata: dict | None = None,
) -> bool:
    """Edit an item's body and/or metadata in place. Returns True if a row changed.

    The store's missing primitive: add/list/resolve/count existed, but not update.
    Both the human (via the app) and the agent (via a future tool) edit through this.
    """
    fields: list[str] = []
    params: list = []
    if body is not None:
        if not body.strip():
            raise ValueError("Empty body.")
        fields.append("body = ?")
        params.append(body.strip())
    if metadata is not None:
        fields.append("metadata = ?")
        params.append(json.dumps(metadata) if metadata else None)
    if not fields:
        return False
    conn = _connect(slug)
    try:
        _ensure_schema(conn)
        params.append(item_id)
        cur = conn.execute(
            f"UPDATE items SET {', '.join(fields)} WHERE id = ?", params
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def count_by_kind(slug: str, status: str | None = "open") -> dict[str, int]:
    conn = _connect(slug)
    try:
        _ensure_schema(conn)
        query = "SELECT kind, COUNT(*) as n FROM items"
        params: list = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " GROUP BY kind"
        rows = conn.execute(query, params).fetchall()
        return {r["kind"]: r["n"] for r in rows}
    finally:
        conn.close()


def _row_to_item(r) -> Item:
    return Item(
        id=r["id"],
        kind=r["kind"],
        body=r["body"],
        source=r["source"],
        source_ref=r["source_ref"],
        status=r["status"],
        created_at=r["created_at"],
        resolved_at=r["resolved_at"],
        metadata=json.loads(r["metadata"]) if r["metadata"] else {},
    )
