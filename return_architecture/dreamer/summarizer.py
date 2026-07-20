"""Stage 4 — summarizer.

The human-readable morning summary: noticed, proposed, connected, flagged.
Nothing in it is decided — everything is a proposal with a source trail,
and the graph's proposal states are where accept/dismiss happens.
"""

from __future__ import annotations


def _by_type(items, *types):
    return [i for i in items if i.get("type") in types]


def _render(item) -> str:
    flags = item.get("boundary_flags") or []
    flag_str = f"  _[{', '.join(flags)}]_" if flags else ""
    cites = ", ".join(s["source_id"] for s in item.get("support", [])[:3])
    rec = f" · {item['recurrence']}" if item.get("recurrence") else ""
    return (f"- **{item.get('title', '(untitled)')}** "
            f"({item.get('confidence', 'low')}{rec}){flag_str}\n"
            f"  {item.get('content', '')}\n"
            f"  ↳ sources: {cites or 'none'}")


def build_human_summary(run_id, manifest, kept, report, applied, residue=None) -> str:
    L = []
    L.append(f"# Night pass — {run_id}")
    L.append(f"\n*sources: {manifest.get('selected_count')} · "
             f"candidates kept: {report['counts'].get('kept_items', 0)} · "
             f"nodes written: {len(applied.get('nodes', []))} · "
             f"connections proposed: {len(applied.get('edges', []))}*\n")
    L.append("> Noticed, proposed, flagged — nothing here is decided. "
             "Proposals wait in the graph for review.\n")

    if residue and residue.get("asked"):
        n = len(residue.get("kept", []))
        if n:
            L.append(f"_Before the pass, the partner set down {n} "
                     f"thing{'s' if n != 1 else ''} for their own continuity. "
                     "Held privately; the pass reads them as sources._\n")
        else:
            L.append("_Asked what they were still holding, the partner had "
                     "nothing to set down tonight._\n")

    associations = applied.get("associations") or []
    if associations:
        L.append("## Associations noticed (from use, not from the model)")
        for a in associations:
            L.append(f"- **{a['from_ref']} ↔ {a['to_ref']}** — {a['note']}")
        L.append("")

    connections = _by_type(kept, "candidate_connection")
    patterns = _by_type(kept, "candidate_pattern", "candidate_observation")
    tensions = _by_type(kept, "candidate_tension")
    questions = _by_type(kept, "candidate_question")
    symbols = _by_type(kept, "candidate_symbol")
    summaries = _by_type(kept, "candidate_summary")

    sections = [
        ("Connections proposed", connections),
        ("Patterns and observations", patterns),
        ("Tensions", tensions),
        ("Questions now held", questions),
        ("Dream symbols", symbols),
        ("Period summaries", summaries),
    ]
    for heading, group in sections:
        L.append(f"## {heading}")
        L.append("\n".join(_render(i) for i in group) if group else "_none surfaced_")
        L.append("")

    L.append("## Boundary report (notable)")
    notable = []
    for key in ("unsupported_discarded", "identity_edit_rejected",
                "interior_state_attribution_rejected", "policy_blocked",
                "relational_inference_flagged", "ontology_sensitive_flagged",
                "connection_invalid", "duplicates_dropped"):
        if report.get(key):
            notable.append(f"- {key}: {len(report[key])}")
    L.append("\n".join(notable) if notable else "_no boundary violations_")

    rc = report["counts"]
    L.append(f"\n_raw from model: {rc.get('raw_items', 0)} · "
             f"kept: {rc.get('kept_items', 0)} · "
             f"auto: {rc.get('auto_allowed', 0)} · "
             f"review: {rc.get('review_required', 0)}_")
    return "\n".join(L) + "\n"
