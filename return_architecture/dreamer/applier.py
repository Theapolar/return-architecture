"""Stage 3.5 — applier.

Writes validated candidates into the memory graph. This stage did not exist
in silas_dream (which wrote JSON queues to folders); here the graph's own
statuses ARE the review queue:

- auto_allowed node candidates   -> nodes with status 'open' (in recall now)
- review_required node candidates-> nodes with status 'proposed' (held out
                                    of recall until accepted)
- connection candidates          -> edges with status 'proposed' (never steer
                                    recall until accepted)
- forbidden candidates           -> never applied; artifacts only

Provenance (`derived_from` edges to the cited sources) is written as accepted
fact even for proposed nodes: that the node was derived from those sources is
true regardless of whether the human keeps the node.
"""

from __future__ import annotations

from return_architecture import graph

_CONF_WEIGHT = {"low": 0.3, "medium": 0.5, "high": 0.7}


def apply(slug: str, kept: list[dict], source_refs: dict[str, str], run_id: str) -> dict:
    applied = {"nodes": [], "edges": [], "skipped": []}
    for item in kept:
        rule = item.get("promotion_rule")
        if rule == "forbidden":
            applied["skipped"].append({"id": item.get("id"), "reason": "forbidden type"})
            continue

        confidence = item.get("confidence", "low")
        weight = _CONF_WEIGHT.get(confidence, 0.3)
        meta = {
            "run_id": run_id,
            "candidate_id": item.get("id"),
            "candidate_type": item.get("type"),
            "confidence": confidence,
            "recurrence": item.get("recurrence", ""),
            "boundary_flags": item.get("boundary_flags", []),
            "promotion_adjustment": item.get("promotion_adjustment", {}),
            "support": item.get("support", []),
        }

        if item.get("type") == "candidate_connection":
            edge_id = graph.add_edge(
                slug,
                from_ref=item["from_ref"],
                to_ref=item["to_ref"],
                rel=item["rel"],
                note=(item.get("content") or item.get("title") or "")[:500],
                author="dreamer",
                weight=weight,
                status="proposed",
            )
            applied["edges"].append({
                "edge_id": edge_id,
                "candidate_id": item.get("id"),
                "from_ref": item["from_ref"],
                "to_ref": item["to_ref"],
                "rel": item["rel"],
                "status": "proposed",
            })
            continue

        node_type = item.get("node_type") or "observation"
        status = "open" if rule == "auto_allowed" else "proposed"
        sources = list(dict.fromkeys(
            source_refs.get(s.get("source_id", ""), "")
            for s in item.get("support", [])
        ))
        sources = [ref for ref in sources if ref]
        node_id = graph.add_node(
            slug,
            type=node_type,
            title=item.get("title", ""),
            body=item.get("content", ""),
            author="dreamer",
            status=status,
            importance=weight,
            metadata=meta,
            sources=sources,
        )
        applied["nodes"].append({
            "node_id": node_id,
            "candidate_id": item.get("id"),
            "type": node_type,
            "status": status,
            "sources": sources,
        })
    return applied
