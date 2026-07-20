"""Stage 1 — collector.

Gathers bounded source material from the agent's world, normalizes it into a
common record format, and runs a STRATIFIED selection (so the dreamer can
notice recurrence across time instead of just paraphrasing the last few
hours). Read-only: the collector never writes anything.

Buckets (ported selection design):
  - structural anchors (commitments / questions / intents / orientation) get a
    small fixed quota and do NOT compete with lived material on recency;
  - summaries (morning notes + previous derived nodes) bridge time;
  - lived notes (captures, dreams, letters, conversation days) carry the day;
  - slots are reserved for OLDER contrast material (2-7d and >7d).

Each record carries a `ref` ("item:N" / "node:N") when it has a home in the
graph, so validated candidates can cite it as a real edge endpoint. Sources
without a graph home (conversation days, letters, orientation) are citable
as evidence but cannot anchor edges.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from return_architecture import graph, items, paths


def iter_ui_turns(slug: str):
    """Every visible conversation turn: the global stream plus all scoped
    thread files. Each turn dict gains a 'thread' key ('' for global)."""
    base = paths.agent_dir(slug)
    files: list[tuple] = []
    global_path = base / "ui_conversation.ndjson"
    if global_path.exists():
        files.append((global_path, ""))
    threads_dir = base / "ui_conversations"
    if threads_dir.exists():
        for p in sorted(threads_dir.iterdir()):
            if p.is_file() and p.suffix == ".ndjson":
                files.append((p, p.stem))
    for path, thread in files:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                turn = json.loads(line)
            except json.JSONDecodeError:
                continue
            turn.setdefault("thread", thread)
            yield turn


def _excerpt(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " …"


def _content_chars(text: str) -> int:
    return len(re.sub(r"\s+", " ", text or "").strip())


def _age_days(now: datetime, ts: str) -> int:
    try:
        then = datetime.fromisoformat(ts)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return max((now - then).days, 0)
    except (ValueError, TypeError):
        return 0


def _record(pool, *, source_id, source_type, bucket, ts, title, text,
            max_excerpt, now, ref="", author="human"):
    pool.append({
        "source_id": source_id,
        "source_type": source_type,
        "ref": ref,
        "timestamp": ts or "",
        "title": (title or "")[:120],
        "author": author,
        "excerpt": _excerpt(text, max_excerpt),
        "bucket": bucket,
        "age_days": _age_days(now, ts or ""),
        "content_chars": _content_chars(text),
    })


_STRUCTURAL_TAGS = {"intent"}
_SUMMARY_TAGS = {"morning_note", "partner_note"}
_SKIP_TAGS = {"canvas", "writing_draft"}


def _gather(slug: str, config: dict, now: datetime) -> list[dict]:
    limits = config["limits"]
    max_excerpt = limits["max_excerpt_chars"]
    window = now - timedelta(days=config["sources"]["days"] * 3)
    pool: list[dict] = []

    # 1. items — the raw layer. Kind + tag decide the bucket.
    for kind in ("note", "important", "question", "commitment"):
        for it in items.list_items(slug, kind=kind, status="open", limit=120):
            meta = it.metadata or {}
            tag = meta.get("tag") or ""
            if tag in _SKIP_TAGS:
                continue
            if kind == "commitment" or kind == "question" or tag in _STRUCTURAL_TAGS:
                bucket = "structural"
            elif tag in _SUMMARY_TAGS:
                bucket = "summary"
            else:
                bucket = "note"
            label = tag or kind
            _record(
                pool,
                source_id=f"item_{it.id}",
                source_type=label,
                bucket=bucket,
                ts=it.created_at,
                title=str(meta.get("title") or "")[:120] or label,
                text=it.body,
                max_excerpt=max_excerpt,
                now=now,
                ref=f"item:{it.id}",
                author=it.source,
            )

    # 2. previous derived nodes — consolidation builds on itself.
    for node in graph.list_nodes(slug, status="open", limit=30):
        if node.author != "human":
            _record(
                pool,
                source_id=f"node_{node.id}",
                source_type=f"derived_{node.type}",
                bucket="summary",
                ts=node.created_at,
                title=node.title or node.type,
                text=node.body,
                max_excerpt=max_excerpt,
                now=now,
                ref=f"node:{node.id}",
                author=node.author,
            )

    # 2b. the partner's held residue — their private end-of-day layer.
    #     Out of recall by status, but first-class evidence for the night.
    for node in graph.list_nodes(slug, status="held", limit=15):
        _record(
            pool,
            source_id=f"residue_{node.id}",
            source_type="residue",
            bucket="note",
            ts=node.created_at,
            title=node.title or "held",
            text=node.body,
            max_excerpt=max_excerpt,
            now=now,
            ref=f"node:{node.id}",
            author=node.author,
        )

    # 3. conversation — one source per recent day (evidence, not an edge end).
    #    All of it: the global stream AND every scoped thread (project/item
    #    windows write to ui_conversations/<thread>.ndjson — a day spent in a
    #    scoped window is still the day).
    by_day: dict[str, list[str]] = {}
    for turn in iter_ui_turns(slug):
        day = str(turn.get("created_at") or "")[:10]
        if not day:
            continue
        # neutral labels: this bundle goes to the third-party night clerk
        who = "human" if turn.get("role") == "you" else "partner"
        text = str(turn.get("text") or "").strip()
        thread = str(turn.get("thread") or "").strip()
        scope = f" (in {thread})" if thread else ""
        if text:
            by_day.setdefault(day, []).append(f"{who}{scope}: {text}")
    for day in sorted(by_day)[-config["sources"]["days"]:]:
        joined = "\n".join(by_day[day])
        _record(
            pool,
            source_id=f"conversation_{day}",
            source_type="conversation_day",
            bucket="note",
            ts=f"{day}T12:00:00+00:00",
            title=f"Conversation, {day}",
            text=joined,
            max_excerpt=max_excerpt * 2,
            now=now,
            author="both",
        )

    # 4. letters — slow considered words, both directions.
    for folder, who in (("inbox", "human"), ("outbox", "partner")):
        d = paths.agent_dir(slug) / folder
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if not (p.is_file() and p.suffix.lower() in {".md", ".txt", ".markdown"}):
                continue
            ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
            if ts < window:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            _record(
                pool,
                source_id=f"letter_{p.stem}"[:60],
                source_type="letter",
                bucket="note",
                ts=ts.isoformat(),
                title=text.splitlines()[0].lstrip("# ").strip()[:120] if text.strip() else p.stem,
                text=text,
                max_excerpt=max_excerpt,
                now=now,
                author=who,
            )

    # 5. orientation — the current "where you are now" (evidence only).
    profile_path = paths.agent_dir(slug) / "profile.json"
    if profile_path.exists():
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            orientation = str(profile.get("orientation") or "").strip()
            if orientation:
                _record(
                    pool,
                    source_id="orientation",
                    source_type="orientation",
                    bucket="structural",
                    ts=now.isoformat(),
                    title="Where you are now",
                    text=orientation,
                    max_excerpt=max_excerpt,
                    now=now,
                    author="partner",
                )
        except (json.JSONDecodeError, OSError):
            pass

    return pool


def _select(pool: list[dict], sel_cfg: dict):
    """Stratified selection, ported: anchors + recency + older contrast."""
    total = sel_cfg["total"]
    quotas = sel_cfg["buckets"]
    min_chars = sel_cfg["min_content_chars"]
    older_windows = sel_cfg["older_contrast_windows_days"]

    for r in pool:
        r["low_value"] = r["content_chars"] < min_chars

    chosen: list[dict] = []
    chosen_ids: set[str] = set()
    trace: dict[str, list[str]] = {}

    def newest(rows):
        return sorted(rows, key=lambda r: r["timestamp"], reverse=True)

    def take(rows, n, label):
        c = 0
        for r in rows:
            if c >= n or len(chosen) >= total:
                break
            if r["source_id"] in chosen_ids:
                continue
            chosen.append(r)
            chosen_ids.add(r["source_id"])
            trace.setdefault(label, []).append(r["source_id"])
            c += 1
        return c

    by_bucket: dict[str, list[dict]] = {}
    for r in pool:
        by_bucket.setdefault(r["bucket"], []).append(r)

    take(newest(by_bucket.get("summary", [])), quotas.get("summary", 2), "summary")
    take(newest(by_bucket.get("structural", [])), quotas.get("structural", 3), "structural")
    notes = [r for r in by_bucket.get("note", []) if not r["low_value"]]
    take(newest(notes), quotas.get("note", 4), "note_recent")

    older_quota = quotas.get("older_contrast", 2)
    lived = [r for r in pool if r["bucket"] in ("note", "summary") and not r["low_value"]]
    per_window = max(1, older_quota // max(1, len(older_windows)))
    for lo, hi in older_windows:
        cands = [r for r in lived
                 if lo <= r["age_days"] <= hi and r["source_id"] not in chosen_ids]
        take(sorted(cands, key=lambda r: -r["content_chars"]), per_window, "older_contrast")

    if len(chosen) < total:
        rest = [r for r in pool if r["source_id"] not in chosen_ids and not r["low_value"]]
        take(newest(rest), total - len(chosen), "fill")
    if len(chosen) < total:
        rest = [r for r in pool if r["source_id"] not in chosen_ids]
        take(newest(rest), total - len(chosen), "fill_lowvalue")

    return chosen, trace


def collect(slug: str, config: dict):
    """Return (manifest, selected_sources, selection_report)."""
    now = datetime.now(timezone.utc)
    pool = _gather(slug, config, now)
    selected, trace = _select(pool, config["selection"])

    def counts_by(rows, key):
        out: dict[str, int] = {}
        for r in rows:
            out[r[key]] = out.get(r[key], 0) + 1
        return out

    report = {
        "created_at": now.isoformat(),
        "pool_size": len(pool),
        "selected_count": len(selected),
        "selected_by_bucket": counts_by(selected, "bucket"),
        "selected_by_type": counts_by(selected, "source_type"),
        "selection_trace": trace,
    }
    public = [
        {k: v for k, v in r.items()
         if k not in ("bucket", "age_days", "content_chars", "low_value")}
        for r in selected
    ]
    manifest = {
        "created_at": now.isoformat(),
        "selected_count": len(public),
        "pool_size": len(pool),
        "limits": config["limits"],
        "policy": config.get("policy", {}),
    }
    return manifest, public, report
