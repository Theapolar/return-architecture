"""Orchestrator — the night pass.

collect -> metabolize -> validate -> apply -> summarize, with every stage's
output written to the run folder under <agent>/dreamer/runs/<run_id>/ so the
whole night is inspectable in the morning.

The model arrives as an injected `complete(system, prompt, max_tokens) -> str`
callable — the caller (the app's rhythm runner, a scheduler, a test) decides
which provider and tier answers. The dreamer itself holds no keys.

Write boundary: this module writes ONLY inside its own run folder and through
the graph's proposal states. It cannot touch items, profile, persona, letters,
or anything else — enforced by _safe_write, not by convention.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from return_architecture import paths

from . import applier, associations, collector, metabolizer, residue, summarizer, validator

DEFAULT_CONFIG = {
    "sources": {"days": 7},
    "associations": {"days": 14, "min_co": 3, "min_lift": 2.0, "min_conf": 0.8, "max_pairs": 3},
    "residue": {"cap": 3, "max_context_chars": 7000},
    "limits": {
        "max_excerpt_chars": 700,
        "max_candidate_items": 8,
        "max_model_tokens": 3000,
    },
    "selection": {
        "total": 12,
        "buckets": {"summary": 2, "structural": 3, "note": 4, "older_contrast": 2},
        "min_content_chars": 80,
        "older_contrast_windows_days": [[2, 7], [8, 99999]],
    },
    "output_caps": {
        "candidate_observation": 2,
        "candidate_pattern": 1,
        "candidate_question": 2,
        "candidate_tension": 1,
        "candidate_summary": 1,
        "candidate_symbol": 1,
        "candidate_connection": 3,
    },
    "policy": {
        "allow_identity_proposals": False,
        "allow_relational_inference": True,
        "allow_ontology_sensitive_items": True,
        "promotion_mode": "proposal_only",
    },
}


def _runs_dir(slug: str) -> Path:
    return paths.agent_dir(slug) / "dreamer" / "runs"


def _safe_write(base: Path, path: Path, text: str) -> None:
    """The write-boundary guard: refuse anything outside the run folder."""
    resolved = path.resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise PermissionError(f"dreamer write outside its boundary: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8")


def run_night_pass(
    slug: str,
    *,
    complete,
    complete_primary=None,
    human_name: str = "",
    partner_name: str = "",
    config: dict | None = None,
) -> dict:
    """Run the full pipeline. Returns a result dict with run_id, counts,
    applied nodes/edges, and the human summary text.

    `complete` is the cheap background tier (metabolizer). `complete_primary`,
    when given, is the partner's own model and voice — used only for the
    residue probe; without it the residue stage is skipped."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    base = _runs_dir(slug)
    run_dir = base / run_id

    def write_json(name: str, obj) -> None:
        _safe_write(base, run_dir / name, json.dumps(obj, indent=2, ensure_ascii=False))

    # --- stage -1: verbalized residue (the partner's own voice, primary tier) ---
    residue_result = {"asked": False, "kept": [], "dropped": []}
    if complete_primary is not None:
        try:
            residue_result = residue.run_residue(
                slug,
                complete=complete_primary,
                human_name=human_name,
                partner_name=partner_name,
                **(cfg.get("residue") or {}),
            )
            _safe_write(base, run_dir / "residue_raw.txt", residue_result.get("raw", ""))
        except Exception as exc:  # the diary must never sink the night
            residue_result = {"asked": False, "kept": [], "dropped": [], "error": str(exc)}
        write_json("residue.json", {k: v for k, v in residue_result.items() if k != "raw"})

    # --- stage 0: co-recall associations (deterministic, model-free) ---
    # Runs before the model stages and regardless of new material: the day's
    # recalls are evidence even when nothing new was captured.
    assoc_cfg = cfg.get("associations") or {}
    try:
        assoc_candidates = associations.mine_associations(slug, **assoc_cfg)
        assoc_applied = associations.propose_associations(slug, assoc_candidates)
    except Exception as exc:  # associations must never sink the night
        assoc_candidates, assoc_applied = [], []
        write_json("associations_error.json", {"error": str(exc)})
    write_json("associations.json", {"candidates": assoc_candidates, "applied": assoc_applied})

    # --- stage 1: collect (read-only) ---
    manifest, selected, selection_report = collector.collect(slug, cfg)
    manifest["run_id"] = run_id
    write_json("input_manifest.json", manifest)
    write_json("selected_sources.json", selected)
    write_json("selection_report.json", selection_report)

    if not selected:
        summary = f"# Night pass {run_id}\n\nNothing in the window; nothing to metabolize.\n"
        _safe_write(base, run_dir / "human_summary.md", summary)
        return {"run_id": run_id, "ok": True,
                "applied": {"nodes": [], "edges": [], "skipped": [], "associations": assoc_applied},
                "residue": residue_result,
                "counts": {"raw_items": 0, "kept_items": 0}, "summary": summary,
                "warning": "no sources in window"}

    # --- stage 2: metabolize ---
    meta = metabolizer.metabolize(
        selected,
        complete=complete,
        human_name=human_name or "the human",
        partner_name=partner_name or "the partner",
        max_items=cfg["limits"]["max_candidate_items"],
        max_tokens=cfg["limits"]["max_model_tokens"],
    )
    write_json("metabolized_items.json", {"items": meta["items"], "error": meta.get("_error")})
    _safe_write(base, run_dir / "model_raw.txt", meta.get("_model_raw", "") or "")
    if meta.get("_error"):
        summary = f"# Night pass {run_id}\n\nModel error: {meta['_error']}\n"
        _safe_write(base, run_dir / "human_summary.md", summary)
        return {"run_id": run_id, "ok": False,
                "applied": {"nodes": [], "edges": [], "skipped": [], "associations": assoc_applied},
                "residue": residue_result,
                "counts": {"raw_items": 0, "kept_items": 0}, "summary": summary,
                "warning": meta["_error"]}

    # --- stage 3: validate (deterministic) ---
    kept, report = validator.validate(
        meta["items"], selected, cfg.get("policy", {}),
        human_name=human_name or "the human",
        partner_name=partner_name or "the partner",
        output_caps=cfg.get("output_caps", {}),
    )
    write_json("boundary_report.json", report)

    # --- stage 3.5: apply into the graph (proposal states) ---
    source_refs = {s["source_id"]: (s.get("ref") or "") for s in selected}
    applied = applier.apply(slug, kept, source_refs, run_id)
    applied["associations"] = assoc_applied
    write_json("applied.json", applied)

    # --- stage 4: summarize ---
    summary = summarizer.build_human_summary(run_id, manifest, kept, report, applied,
                                             residue=residue_result)
    _safe_write(base, run_dir / "human_summary.md", summary)

    return {
        "run_id": run_id,
        "ok": True,
        "counts": report["counts"],
        "applied": applied,
        "residue": residue_result,
        "summary": summary,
    }
