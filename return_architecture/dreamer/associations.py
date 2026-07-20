"""Co-recall associations — "fire together, wire together."

The deterministic association stage of the night pass (docs/workspace.md,
mechanism 1). The retrieval log records which refs entered context together;
refs that keep co-surfacing beyond chance are likely related in the sense
the workspace paper means — held simultaneously, repeatedly.

This stage involves no model call and therefore cannot hallucinate an
association: every proposal is a counting fact about actual use, cited as
such in the edge note. Proposals flow through the same review gate as
everything else the dreamer produces (proposed edges; recall ignores them
until accepted; a let-go is remembered and never re-proposed).

Deliberate exclusions:
- entries that arrived in a recall via edge expansion (via='edge:*') do not
  count — associations must form from independent co-surfacing, or accepted
  edges would breed more edges (rich-get-richer feedback);
- pairs that already have ANY edge row between them, in any status —
  including 'rejected', so a human's let-go is final;
- test/verification surfaces.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import combinations

from return_architecture import graph

_SKIP_SURFACES = ("verify", "test", "golden")
_CONF_WEIGHT_MIN, _CONF_WEIGHT_MAX = 0.4, 0.7


def _recall_sets(slug: str, days: int) -> list[dict]:
    """Recall sets from the trailing window: [{'refs': set, 'date': 'YYYY-MM-DD'}]."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    sets: list[dict] = []
    for entry in graph.retrieval_log(slug, limit=1000):
        surface = str(entry.get("surface") or "")
        if any(surface.startswith(s) or surface == s for s in _SKIP_SURFACES):
            continue
        try:
            ts = datetime.fromisoformat(entry["ts"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        if ts < cutoff:
            continue
        refs = {
            r["ref"] for r in entry.get("returned") or []
            if not str(r.get("via") or "").startswith("edge:")
        }
        if len(refs) >= 2:
            sets.append({"refs": refs, "date": entry["ts"][:10]})
    return sets


def _pair_has_edge(slug: str, a: str, b: str) -> bool:
    return any(
        edge.to_ref == b or edge.from_ref == b
        for edge in graph.edges_for(slug, a)
    )


def mine_associations(
    slug: str,
    *,
    days: int = 14,
    min_co: int = 3,
    min_lift: float = 2.0,
    min_conf: float = 0.8,
    max_pairs: int = 3,
) -> list[dict]:
    """Ranked association candidates from the retrieval log. Read-only.

    A pair qualifies by either route:
    - lift >= min_lift: co-surfaces more than its members' frequencies predict
      (discriminates in dense logs);
    - confidence >= min_conf: co-occurs in nearly every appearance of its rarer
      member (catches perfect pairings in sparse logs, where lift saturates
      at 1.0 because both members appear in every set).
    """
    sets = _recall_sets(slug, days)
    n = len(sets)
    if n == 0:
        return []

    ref_count: dict[str, int] = {}
    pair_count: dict[tuple[str, str], int] = {}
    pair_dates: dict[tuple[str, str], set[str]] = {}
    for s in sets:
        for ref in s["refs"]:
            ref_count[ref] = ref_count.get(ref, 0) + 1
        for a, b in combinations(sorted(s["refs"]), 2):
            pair_count[(a, b)] = pair_count.get((a, b), 0) + 1
            pair_dates.setdefault((a, b), set()).add(s["date"])

    candidates: list[dict] = []
    for (a, b), co in pair_count.items():
        if co < min_co:
            continue
        # lift: how much more often the pair co-surfaces than its members'
        # individual frequencies predict under independence
        lift = (co * n) / (ref_count[a] * ref_count[b])
        # confidence: share of the rarer member's appearances that include the other
        conf = co / min(ref_count[a], ref_count[b])
        if lift < min_lift and conf < min_conf:
            continue
        if _pair_has_edge(slug, a, b):
            continue
        entry_a = graph.entry_for_ref(slug, a)
        entry_b = graph.entry_for_ref(slug, b)
        if entry_a is None or entry_b is None:
            continue
        candidates.append({
            "from_ref": a,
            "to_ref": b,
            "co_count": co,
            "days_span": len(pair_dates[(a, b)]),
            "lift": round(lift, 2),
            "confidence": round(conf, 2),
            "strength": co * max(lift, conf),
            "labels": [
                (entry_a.title or entry_a.body)[:90],
                (entry_b.title or entry_b.body)[:90],
            ],
        })

    candidates.sort(key=lambda c: c["strength"], reverse=True)
    return candidates[:max_pairs]


def propose_associations(slug: str, candidates: list[dict]) -> list[dict]:
    """Write candidates as proposed edges. Returns what was applied."""
    applied: list[dict] = []
    for cand in candidates:
        days = cand["days_span"]
        note = (
            f"Surfaced together in recall {cand['co_count']} times"
            + (f" across {days} days" if days > 1 else " in one day")
            + " — they may belong together."
        )
        weight = min(_CONF_WEIGHT_MAX, _CONF_WEIGHT_MIN + 0.05 * cand["co_count"])
        edge_id = graph.add_edge(
            slug,
            from_ref=cand["from_ref"],
            to_ref=cand["to_ref"],
            rel="relates",
            note=note,
            author="dreamer",
            weight=weight,
            status="proposed",
        )
        applied.append({
            "edge_id": edge_id,
            "from_ref": cand["from_ref"],
            "to_ref": cand["to_ref"],
            "co_count": cand["co_count"],
            "days_span": cand["days_span"],
            "lift": cand["lift"],
            "note": note,
            "status": "proposed",
        })
    return applied
