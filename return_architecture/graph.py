"""The memory graph: derived nodes, edges, search, budgeted recall.

This is the connective layer of the memory system (ARCHITECTURE.md "The
memory system (the heart)"). It lives in the same per-agent sqlite file as
items.db and treats the existing `items` table as the raw-capture layer:

- raw captures      -> items.py (untouched; the dreamer never rewrites them)
- derived nodes     -> `nodes` table here (summaries, patterns, symbols,
                       observations the consolidation pass writes)
- edges             -> `edges` table, linking any two refs
- retrieval         -> FTS5 over items + nodes, packed to a budget
- retrieval log     -> `retrievals` table (inspectability: what was recalled,
                       when, for which surface, and why)
- touch signals     -> `touches` table (how often a ref is recalled; one of
                       the computed-importance inputs, alongside recency)

Refs are strings like "item:42" or "node:7" — an open vocabulary, so later
layers (dreams, letters) can join the graph without schema changes.

Design rules this module enforces or supports:
- Raw and derived stay separate; derived nodes cite sources via
  `derived_from` edges rather than replacing anything.
- Human tags remain freeform (items metadata); the *system's* vocabulary
  (node types, edge rels) is the small controlled set below. The store is
  permissive — the dreamer's validator is where strictness lives — but
  everything the system authors should come from these sets.
- Every recall is logged. Silence about what memory did is not allowed.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from return_architecture import paths

# The system's controlled vocabularies. The dreamer may merge/prune within
# these; human-authored tags elsewhere are never constrained by them.
NODE_TYPES: tuple[str, ...] = (
    "summary",       # compression of a period or cluster
    "pattern",       # something that keeps happening
    "observation",   # a single noticed thing, cited
    "symbol",        # recurring dream symbol (the constellation)
    "question",      # something the system is holding open
    "tension",       # two things that pull against each other
)

EDGE_RELS: tuple[str, ...] = (
    "relates",       # generic association
    "supports",      # evidence for
    "supersedes",    # from_ref replaces to_ref as current truth
    "derived_from",  # derived node -> its raw sources
    "recurs_in",     # symbol -> dream, pattern -> instance
    "serves",        # commitment -> intent, work -> aim
)

_REF_RE = re.compile(r"^[a-z_]+:[0-9a-zA-Z_-]+$")


@dataclass
class Node:
    id: int
    type: str
    title: str
    body: str
    author: str
    status: str
    importance: float
    created_at: str
    updated_at: str
    metadata: dict


@dataclass
class Edge:
    id: int
    from_ref: str
    to_ref: str
    rel: str
    note: str
    author: str
    weight: float
    status: str
    created_at: str
    updated_at: str


@dataclass
class RecallEntry:
    ref: str
    kind: str        # item kind or node type
    title: str       # short label ('' for items)
    body: str
    score: float
    created_at: str
    via: str         # 'match' (search hit) or 'edge:<rel>' (graph neighbor)


@dataclass
class RecallResult:
    entries: list[RecallEntry] = field(default_factory=list)
    text: str = ""           # the packed context block, ready for a prompt
    budget: int = 0
    used: int = 0
    log_id: int | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db_path(slug: str) -> Path:
    return paths.agent_dir(slug) / "items.db"


def _connect(slug: str) -> sqlite3.Connection:
    paths.agent_dir(slug).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_db_path(slug))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
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
        );

        CREATE TABLE IF NOT EXISTS nodes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            type        TEXT NOT NULL,
            title       TEXT NOT NULL DEFAULT '',
            body        TEXT NOT NULL,
            author      TEXT NOT NULL DEFAULT 'dreamer',
            status      TEXT NOT NULL DEFAULT 'open',
            importance  REAL NOT NULL DEFAULT 0.5,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            metadata    TEXT
        );

        CREATE TABLE IF NOT EXISTS edges (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            from_ref    TEXT NOT NULL,
            to_ref      TEXT NOT NULL,
            rel         TEXT NOT NULL,
            note        TEXT NOT NULL DEFAULT '',
            author      TEXT NOT NULL DEFAULT 'dreamer',
            weight      REAL NOT NULL DEFAULT 0.5,
            status      TEXT NOT NULL DEFAULT 'accepted',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            UNIQUE(from_ref, to_ref, rel)
        );
        CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_ref);
        CREATE INDEX IF NOT EXISTS idx_edges_to   ON edges(to_ref);

        CREATE TABLE IF NOT EXISTS touches (
            ref      TEXT PRIMARY KEY,
            count    INTEGER NOT NULL DEFAULT 0,
            last_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS retrievals (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            ts       TEXT NOT NULL,
            surface  TEXT NOT NULL DEFAULT '',
            query    TEXT NOT NULL DEFAULT '',
            budget   INTEGER NOT NULL DEFAULT 0,
            used     INTEGER NOT NULL DEFAULT 0,
            returned TEXT
        );
        """
    )
    # FTS over raw items, kept in sync by triggers. External-content tables
    # stay small (index only) and rebuild cleanly if ever out of step.
    fts_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='items_fts'"
    ).fetchone()
    conn.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS items_fts
            USING fts5(body, content='items', content_rowid='id');
        CREATE TRIGGER IF NOT EXISTS items_fts_ai AFTER INSERT ON items BEGIN
            INSERT INTO items_fts(rowid, body) VALUES (new.id, new.body);
        END;
        CREATE TRIGGER IF NOT EXISTS items_fts_ad AFTER DELETE ON items BEGIN
            INSERT INTO items_fts(items_fts, rowid, body) VALUES ('delete', old.id, old.body);
        END;
        CREATE TRIGGER IF NOT EXISTS items_fts_au AFTER UPDATE OF body ON items BEGIN
            INSERT INTO items_fts(items_fts, rowid, body) VALUES ('delete', old.id, old.body);
            INSERT INTO items_fts(rowid, body) VALUES (new.id, new.body);
        END;

        CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
            USING fts5(title, body, content='nodes', content_rowid='id');
        CREATE TRIGGER IF NOT EXISTS nodes_fts_ai AFTER INSERT ON nodes BEGIN
            INSERT INTO nodes_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
        END;
        CREATE TRIGGER IF NOT EXISTS nodes_fts_ad AFTER DELETE ON nodes BEGIN
            INSERT INTO nodes_fts(nodes_fts, rowid, title, body) VALUES ('delete', old.id, old.title, old.body);
        END;
        CREATE TRIGGER IF NOT EXISTS nodes_fts_au AFTER UPDATE OF title, body ON nodes BEGIN
            INSERT INTO nodes_fts(nodes_fts, rowid, title, body) VALUES ('delete', old.id, old.title, old.body);
            INSERT INTO nodes_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
        END;
        """
    )
    if not fts_exists:
        # First time here on a database with pre-existing items: index them.
        conn.execute("INSERT INTO items_fts(items_fts) VALUES ('rebuild')")
    # Migration: edges created before review existed lack the status column.
    edge_cols = {r["name"] for r in conn.execute("PRAGMA table_info(edges)").fetchall()}
    if "status" not in edge_cols:
        conn.execute("ALTER TABLE edges ADD COLUMN status TEXT NOT NULL DEFAULT 'accepted'")
    conn.commit()


# ---------------------------------------------------------------------------
# nodes — the derived layer
# ---------------------------------------------------------------------------
def add_node(
    slug: str,
    *,
    type: str,
    body: str,
    title: str = "",
    author: str = "dreamer",
    status: str = "open",
    importance: float = 0.5,
    metadata: dict | None = None,
    sources: list[str] | None = None,
) -> int:
    """Create a derived node. `sources` are refs the node was derived from;
    they become `derived_from` edges so provenance is queryable.
    status='proposed' keeps a node out of recall until it is accepted."""
    if not body.strip():
        raise ValueError("Empty body.")
    conn = _connect(slug)
    try:
        now = _now()
        cur = conn.execute(
            "INSERT INTO nodes (type, title, body, author, status, importance, created_at, updated_at, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                type.strip(), title.strip(), body.strip(), author, status,
                float(importance), now, now,
                json.dumps(metadata) if metadata else None,
            ),
        )
        conn.commit()
        node_id = cur.lastrowid or 0
    finally:
        conn.close()
    for source in sources or []:
        add_edge(slug, from_ref=f"node:{node_id}", to_ref=source, rel="derived_from", author=author)
    return node_id


def get_node(slug: str, node_id: int) -> Node | None:
    conn = _connect(slug)
    try:
        row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
        return _row_to_node(row) if row else None
    finally:
        conn.close()


def list_nodes(
    slug: str,
    *,
    type: str | None = None,
    status: str | None = "open",
    limit: int = 50,
) -> list[Node]:
    conn = _connect(slug)
    try:
        query = "SELECT * FROM nodes WHERE 1=1"
        params: list = []
        if type is not None:
            query += " AND type = ?"
            params.append(type)
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(limit)
        return [_row_to_node(r) for r in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


def update_node(
    slug: str,
    node_id: int,
    *,
    title: str | None = None,
    body: str | None = None,
    status: str | None = None,
    importance: float | None = None,
    metadata: dict | None = None,
) -> bool:
    fields: list[str] = []
    params: list = []
    if title is not None:
        fields.append("title = ?"); params.append(title.strip())
    if body is not None:
        if not body.strip():
            raise ValueError("Empty body.")
        fields.append("body = ?"); params.append(body.strip())
    if status is not None:
        fields.append("status = ?"); params.append(status)
    if importance is not None:
        fields.append("importance = ?"); params.append(float(importance))
    if metadata is not None:
        fields.append("metadata = ?"); params.append(json.dumps(metadata) if metadata else None)
    if not fields:
        return False
    fields.append("updated_at = ?"); params.append(_now())
    conn = _connect(slug)
    try:
        params.append(node_id)
        cur = conn.execute(f"UPDATE nodes SET {', '.join(fields)} WHERE id = ?", params)
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def supersede_node(slug: str, new_id: int, old_id: int, note: str = "", author: str = "dreamer") -> bool:
    """Mark old as superseded by new — the graph's answer to contradiction.
    The old node stays (inspectable history) but leaves the default recall set."""
    changed = update_node(slug, old_id, status="superseded")
    if changed:
        add_edge(slug, from_ref=f"node:{new_id}", to_ref=f"node:{old_id}", rel="supersedes", note=note, author=author)
    return changed


# ---------------------------------------------------------------------------
# edges — the connective tissue
# ---------------------------------------------------------------------------
def add_edge(
    slug: str,
    *,
    from_ref: str,
    to_ref: str,
    rel: str = "relates",
    note: str = "",
    author: str = "dreamer",
    weight: float = 0.5,
    status: str = "accepted",
) -> int:
    """Upsert an edge (unique per from/to/rel). Returns the edge id.
    status='proposed' keeps an edge out of recall expansion until accepted;
    an upsert never downgrades an accepted edge back to proposed."""
    for ref in (from_ref, to_ref):
        if not _REF_RE.match(ref or ""):
            raise ValueError(f"Bad ref '{ref}' (expected e.g. 'item:42' or 'node:7').")
    if from_ref == to_ref:
        raise ValueError("An edge needs two different ends.")
    conn = _connect(slug)
    try:
        now = _now()
        conn.execute(
            "INSERT INTO edges (from_ref, to_ref, rel, note, author, weight, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(from_ref, to_ref, rel) DO UPDATE SET "
            "note = CASE WHEN excluded.note != '' THEN excluded.note ELSE note END, "
            "weight = excluded.weight, "
            "status = CASE WHEN status = 'accepted' THEN 'accepted' ELSE excluded.status END, "
            "updated_at = excluded.updated_at",
            (from_ref, to_ref, rel, note.strip(), author, float(weight), status, now, now),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM edges WHERE from_ref = ? AND to_ref = ? AND rel = ?",
            (from_ref, to_ref, rel),
        ).fetchone()
        return int(row["id"]) if row else 0
    finally:
        conn.close()


def edges_for(slug: str, ref: str, rel: str | None = None) -> list[Edge]:
    conn = _connect(slug)
    try:
        query = "SELECT * FROM edges WHERE (from_ref = ? OR to_ref = ?)"
        params: list = [ref, ref]
        if rel is not None:
            query += " AND rel = ?"
            params.append(rel)
        query += " ORDER BY weight DESC, updated_at DESC"
        return [_row_to_edge(r) for r in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


def list_edges(
    slug: str,
    *,
    status: str | None = None,
    author: str | None = None,
    rel: str | None = None,
    limit: int = 50,
) -> list[Edge]:
    conn = _connect(slug)
    try:
        query = "SELECT * FROM edges WHERE 1=1"
        params: list = []
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        if author is not None:
            query += " AND author = ?"
            params.append(author)
        if rel is not None:
            query += " AND rel = ?"
            params.append(rel)
        query += " ORDER BY weight DESC, updated_at DESC LIMIT ?"
        params.append(limit)
        return [_row_to_edge(r) for r in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


def remove_edge(slug: str, edge_id: int) -> bool:
    conn = _connect(slug)
    try:
        cur = conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def neighbors(slug: str, ref: str, limit: int = 12) -> list[tuple[str, Edge]]:
    """Refs one hop away along ACCEPTED edges, strongest first. Proposed and
    rejected edges never steer recall."""
    out: list[tuple[str, Edge]] = []
    for edge in edges_for(slug, ref):
        if edge.status != "accepted":
            continue
        other = edge.to_ref if edge.from_ref == ref else edge.from_ref
        out.append((other, edge))
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# search + recall
# ---------------------------------------------------------------------------
def _fts_query(raw: str) -> str:
    """Turn free text into a safe FTS5 OR-query: any term may match, ranked."""
    tokens = re.findall(r"[\w']+", raw or "", flags=re.UNICODE)
    return " OR ".join(f'"{t}"' for t in tokens[:16])


def search(slug: str, query: str, limit: int = 12) -> list[RecallEntry]:
    """Ranked full-text hits across raw items and derived nodes."""
    fts = _fts_query(query)
    if not fts:
        return []
    conn = _connect(slug)
    try:
        entries: list[RecallEntry] = []
        for row in conn.execute(
            "SELECT i.id, i.kind, i.body, i.created_at, i.metadata, bm25(items_fts) AS rank "
            "FROM items_fts JOIN items i ON i.id = items_fts.rowid "
            "WHERE items_fts MATCH ? AND i.status = 'open' "
            "ORDER BY rank LIMIT ?",
            (fts, limit),
        ).fetchall():
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            entries.append(RecallEntry(
                ref=f"item:{row['id']}",
                kind=str(meta.get("tag") or row["kind"]),
                title=str(meta.get("title") or ""),
                body=row["body"],
                score=-float(row["rank"]),   # bm25 is smaller-is-better; flip it
                created_at=row["created_at"],
                via="match",
            ))
        for row in conn.execute(
            "SELECT n.id, n.type, n.title, n.body, n.created_at, n.importance, bm25(nodes_fts) AS rank "
            "FROM nodes_fts JOIN nodes n ON n.id = nodes_fts.rowid "
            "WHERE nodes_fts MATCH ? AND n.status = 'open' "
            "ORDER BY rank LIMIT ?",
            (fts, limit),
        ).fetchall():
            entries.append(RecallEntry(
                ref=f"node:{row['id']}",
                kind=row["type"],
                title=row["title"],
                body=row["body"],
                # derived nodes carry their computed importance into ranking
                score=-float(row["rank"]) + float(row["importance"]),
                created_at=row["created_at"],
                via="match",
            ))
        entries.sort(key=lambda e: e.score, reverse=True)
        return entries[:limit]
    finally:
        conn.close()


def _entry_for_ref(conn: sqlite3.Connection, ref: str, via: str) -> RecallEntry | None:
    kind_name, _, raw_id = ref.partition(":")
    if not raw_id.isdigit():
        return None
    if kind_name == "item":
        row = conn.execute(
            "SELECT * FROM items WHERE id = ? AND status = 'open'", (int(raw_id),)
        ).fetchone()
        if not row:
            return None
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return RecallEntry(
            ref=ref, kind=str(meta.get("tag") or row["kind"]),
            title=str(meta.get("title") or ""), body=row["body"],
            score=0.0, created_at=row["created_at"], via=via,
        )
    if kind_name == "node":
        row = conn.execute(
            "SELECT * FROM nodes WHERE id = ? AND status = 'open'", (int(raw_id),)
        ).fetchone()
        if not row:
            return None
        return RecallEntry(
            ref=ref, kind=row["type"], title=row["title"], body=row["body"],
            score=float(row["importance"]), created_at=row["created_at"], via=via,
        )
    return None


def _touch(conn: sqlite3.Connection, refs: list[str]) -> None:
    now = _now()
    for ref in refs:
        conn.execute(
            "INSERT INTO touches (ref, count, last_at) VALUES (?, 1, ?) "
            "ON CONFLICT(ref) DO UPDATE SET count = count + 1, last_at = ?",
            (ref, now, now),
        )


def _render(entries: list[RecallEntry]) -> str:
    lines = []
    for e in entries:
        label = e.kind.capitalize() if e.kind else "Note"
        title = f" {e.title} —" if e.title else ""
        body = re.sub(r"\s+", " ", e.body).strip()
        lines.append(f"- [{label} {e.ref}]{title} {body}")
    return "\n".join(lines)


def recall(
    slug: str,
    query: str,
    *,
    budget: int = 2000,
    surface: str = "",
    limit: int = 10,
    expand: bool = True,
) -> RecallResult:
    """The recall primitive every surface should call.

    Full-text hits across items + nodes, expanded one hop along the graph,
    packed into `budget` characters, logged to `retrievals`, and touch-counted.
    With an empty query, returns the most recent open material (the morning's
    "what is alive right now" mode).
    """
    hits = search(slug, query, limit=limit) if query.strip() else _recent_entries(slug, limit)

    conn = _connect(slug)
    try:
        # one hop out along the strongest edges of the top hits
        if expand:
            have = {e.ref for e in hits}
            for hit in list(hits)[:5]:
                for other_ref, edge in neighbors(slug, hit.ref, limit=4):
                    if other_ref in have:
                        continue
                    entry = _entry_for_ref(conn, other_ref, via=f"edge:{edge.rel}")
                    if entry is not None:
                        have.add(other_ref)
                        hits.append(entry)

        picked: list[RecallEntry] = []
        used = 0
        for entry in hits:
            cost = len(entry.body) + len(entry.title) + 24
            if picked and used + cost > budget:
                continue
            picked.append(entry)
            used += cost
            if used >= budget:
                break

        _touch(conn, [e.ref for e in picked])
        cur = conn.execute(
            "INSERT INTO retrievals (ts, surface, query, budget, used, returned) VALUES (?, ?, ?, ?, ?, ?)",
            (
                _now(), surface, query, budget, used,
                json.dumps([{"ref": e.ref, "score": round(e.score, 3), "via": e.via} for e in picked]),
            ),
        )
        conn.commit()
        return RecallResult(entries=picked, text=_render(picked), budget=budget, used=used, log_id=cur.lastrowid)
    finally:
        conn.close()


def _recent_entries(slug: str, limit: int) -> list[RecallEntry]:
    conn = _connect(slug)
    try:
        entries: list[RecallEntry] = []
        for row in conn.execute(
            "SELECT * FROM items WHERE status = 'open' ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall():
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            entries.append(RecallEntry(
                ref=f"item:{row['id']}", kind=str(meta.get("tag") or row["kind"]),
                title=str(meta.get("title") or ""), body=row["body"],
                score=0.0, created_at=row["created_at"], via="recent",
            ))
        for row in conn.execute(
            "SELECT * FROM nodes WHERE status = 'open' ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall():
            entries.append(RecallEntry(
                ref=f"node:{row['id']}", kind=row["type"], title=row["title"], body=row["body"],
                score=float(row["importance"]), created_at=row["created_at"], via="recent",
            ))
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries[:limit]
    finally:
        conn.close()


def entry_for_ref(slug: str, ref: str) -> RecallEntry | None:
    """Public lookup of a single ref — used to label edge ends in review UIs."""
    conn = _connect(slug)
    try:
        return _entry_for_ref(conn, ref, via="lookup")
    finally:
        conn.close()


def retrieval_log(slug: str, limit: int = 30) -> list[dict]:
    conn = _connect(slug)
    try:
        out = []
        for row in conn.execute(
            "SELECT * FROM retrievals ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall():
            out.append({
                "id": row["id"], "ts": row["ts"], "surface": row["surface"],
                "query": row["query"], "budget": row["budget"], "used": row["used"],
                "returned": json.loads(row["returned"]) if row["returned"] else [],
            })
        return out
    finally:
        conn.close()


def touch_counts(slug: str, refs: list[str] | None = None) -> dict[str, dict]:
    conn = _connect(slug)
    try:
        if refs:
            marks = ",".join("?" for _ in refs)
            rows = conn.execute(f"SELECT * FROM touches WHERE ref IN ({marks})", refs).fetchall()
        else:
            rows = conn.execute("SELECT * FROM touches ORDER BY count DESC LIMIT 100").fetchall()
        return {r["ref"]: {"count": r["count"], "last_at": r["last_at"]} for r in rows}
    finally:
        conn.close()


def _row_to_node(r) -> Node:
    return Node(
        id=r["id"], type=r["type"], title=r["title"], body=r["body"],
        author=r["author"], status=r["status"], importance=r["importance"],
        created_at=r["created_at"], updated_at=r["updated_at"],
        metadata=json.loads(r["metadata"]) if r["metadata"] else {},
    )


def _row_to_edge(r) -> Edge:
    return Edge(
        id=r["id"], from_ref=r["from_ref"], to_ref=r["to_ref"], rel=r["rel"],
        note=r["note"], author=r["author"], weight=r["weight"], status=r["status"],
        created_at=r["created_at"], updated_at=r["updated_at"],
    )


# ---------------------------------------------------------------------------
# proposals — the dreamer proposes; the human decides
# ---------------------------------------------------------------------------
def list_proposals(slug: str, limit: int = 50) -> dict:
    """Everything awaiting review: proposed nodes and proposed edges."""
    conn = _connect(slug)
    try:
        nodes = [
            _row_to_node(r) for r in conn.execute(
                "SELECT * FROM nodes WHERE status = 'proposed' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        ]
        edges = [
            _row_to_edge(r) for r in conn.execute(
                "SELECT * FROM edges WHERE status = 'proposed' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        ]
        return {"nodes": nodes, "edges": edges}
    finally:
        conn.close()


def resolve_node_proposal(slug: str, node_id: int, accept: bool) -> bool:
    """Accept opens the node into recall; dismiss keeps it, inspectable, as rejected."""
    conn = _connect(slug)
    try:
        cur = conn.execute(
            "UPDATE nodes SET status = ?, updated_at = ? WHERE id = ? AND status = 'proposed'",
            ("open" if accept else "rejected", _now(), node_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def resolve_edge_proposal(slug: str, edge_id: int, accept: bool) -> bool:
    conn = _connect(slug)
    try:
        cur = conn.execute(
            "UPDATE edges SET status = ?, updated_at = ? WHERE id = ? AND status = 'proposed'",
            ("accepted" if accept else "rejected", _now(), edge_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()
