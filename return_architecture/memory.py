"""Per-agent semantic memory backed by ChromaDB.

One collection per agent, stored at <agent>/memory/. Uses Chroma's default
embedding model (all-MiniLM-L6-v2 via ONNX), which downloads on first use
and runs entirely locally thereafter.

The memory layer is intentionally thin: store user and assistant turns as
they happen, retrieve top-K semantically similar past turns before each
new user message. Higher-level constructs (summaries, sessions, tagged
items) are separate concerns.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import chromadb
from chromadb.config import Settings

from return_architecture import paths


@dataclass
class MemoryEntry:
    content: str
    role: str
    timestamp: str
    session_id: str
    distance: float | None = None


class MemoryStore:
    def __init__(self, slug: str) -> None:
        self._slug = slug
        memory_dir = paths.agent_dir(slug) / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(memory_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name="memory")

    def remember(self, content: str, *, role: str, session_id: str) -> None:
        if not content or not content.strip():
            return
        self._collection.add(
            ids=[str(uuid.uuid4())],
            documents=[content],
            metadatas=[{
                "role": role,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
            }],
        )

    def recall(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        count = self._collection.count()
        if count == 0 or not query.strip():
            return []
        result = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
        )
        out: list[MemoryEntry] = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            out.append(MemoryEntry(
                content=doc,
                role=str(meta.get("role", "")),
                timestamp=str(meta.get("timestamp", "")),
                session_id=str(meta.get("session_id", "")),
                distance=float(dist),
            ))
        return out

    def recent(self, limit: int = 20) -> list[MemoryEntry]:
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.get(limit=count)
        entries = [
            MemoryEntry(
                content=doc,
                role=str(meta.get("role", "")),
                timestamp=str(meta.get("timestamp", "")),
                session_id=str(meta.get("session_id", "")),
            )
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries[:limit]

    def count(self) -> int:
        return self._collection.count()


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]
