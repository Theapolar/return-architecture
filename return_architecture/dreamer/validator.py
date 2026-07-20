"""Stage 3 — validator.

The deterministic boundary layer, ported from silas_dream with its rules
intact and its person-specific regexes parameterized. This is the real safety
mechanism: it does NOT trust the model's self-assigned confidence or
promotion rules. It re-derives promotion from type + boundary flags, computes
recurrence from the citations (never the model's claim), checks
support-alignment (catches citation drift), blocks interior-state attribution
(no mind-reading), constrains ontology-sensitive topics to
questions/tensions, dedupes, and caps output.

New here relative to silas_dream: candidate_connection — a proposed edge
between two cited sources. The validator additionally requires that both
ends resolve to known sources that HAVE graph refs, that the rel is in the
system vocabulary, and that the ends differ.
"""

from __future__ import annotations

import re

VALID_TYPES = {
    "candidate_observation",
    "candidate_pattern",
    "candidate_question",
    "candidate_tension",
    "candidate_summary",
    "candidate_symbol",
    "candidate_connection",
    "candidate_commitment",   # recognized so it can be deterministically forbidden
}

# type -> graph node type (connections become edges instead)
TYPE_NODE = {
    "candidate_observation": "observation",
    "candidate_pattern": "pattern",
    "candidate_question": "question",
    "candidate_tension": "tension",
    "candidate_summary": "summary",
    "candidate_symbol": "symbol",
}

# Default promotion rule per type. Only questions walk in unreviewed; the
# dreamer may never author commitments at all.
TYPE_DEFAULT_RULE = {
    "candidate_question": "auto_allowed",
    "candidate_observation": "review_required",
    "candidate_pattern": "review_required",
    "candidate_tension": "review_required",
    "candidate_summary": "review_required",
    "candidate_symbol": "review_required",
    "candidate_connection": "review_required",
    "candidate_commitment": "forbidden",
}

ONTOLOGY_ALLOWED_TYPES = {"candidate_question", "candidate_tension"}

EDGE_RELS_ALLOWED = {"relates", "supports", "serves", "recurs_in", "supersedes"}

_ONTOLOGY_RE = re.compile(
    r"\b(conscious|consciousness|sentien|experience|qualia|selfhood|"
    r"personhood|feel(s|ing)?|alive|soul|aware(ness)?|inner life|"
    r"moral status|suffer)\b", re.I)

_IDENTITY_RE = re.compile(
    r"\b(identity|manifesto|who I (really )?am|my true self|core self|"
    r"rewrite (the )?identity)\b", re.I)

_INFLATION_RE = re.compile(
    r"\b(always|never|the true|definitely|proves?|clearly is|this is who)\b", re.I)

_INTERIOR_VERBS = (
    r"(want|wants|wanted|need|needs|needed|feel|feels|felt|fear|fears|feared|"
    r"hope|hopes|hoped|wish|wishes|wished|believe|believes|believed|think|"
    r"thinks|thought|love|loves|loved|hate|hates|hated|knew|knows|intend|"
    r"intends|intended|mean|means|meant|desire|desires|desired|motive|"
    r"feeling|feelings|intention|intentions)"
)

_SELF_STANCE_VERBS = (
    r"(is|was|believes|believed|experiences|experienced|feels|felt|knows|"
    r"possesses|has\s+(consciousness|experience|qualia|a\s+self|selfhood|"
    r"inner\s+life))"
)

_OBSERVABLE_RE = re.compile(
    r"\b(across\s+\d+|returns?\s+to|recurs?|recurring|appears?\s+in|repeated|"
    r"repeatedly|\d+\s+sources|multiple\s+(notes|sources|places)|"
    r"distinction\s+between)\b", re.I)


def _person_res(human_name: str, partner_name: str):
    """Build the person-keyed guards for THIS human and THIS partner."""
    h = re.escape((human_name or "the human").strip() or "the human")
    p = re.escape((partner_name or "the partner").strip() or "the partner")
    interior_other = re.compile(rf"\b{h}(?:'s)?\s+{_INTERIOR_VERBS}\b", re.I)
    self_stance = re.compile(rf"\b{p}\s+{_SELF_STANCE_VERBS}\b", re.I)
    human_mention = re.compile(rf"\b{h}\b", re.I)
    return interior_other, self_stance, human_mention


def _norm_rule(val):
    v = str(val or "").lower()
    return v if v in ("auto_allowed", "review_required", "forbidden") else None


def _norm_confidence(val) -> str:
    v = str(val or "").lower()
    return v if v in ("low", "medium", "high") else "low"


_CONF_ORDER = {"high": 0, "medium": 1, "low": 2}


def _has_quoted_support(norm_support) -> bool:
    return any('"' in s["excerpt"] or "“" in s["excerpt"] or "”" in s["excerpt"]
               for s in norm_support)


_STOP = set(
    "the a an and or but if then this that these those of to in on for with as "
    "is are was were be been being it its his her their my your our we you they "
    "i he she them us about into over under not no nor so than too very can will "
    "just only also more most some any each what does look like would could "
    "should from at by out up down".split())

_WORD_RE = re.compile(r"[a-z][a-z'\-]{2,}")


def _content_tokens(text: str) -> set:
    return {t for t in _WORD_RE.findall((text or "").lower())
            if t not in _STOP and len(t) >= 4}


def _support_alignment(title: str, content: str, norm_support):
    """Best lexical overlap between the claim and any one cited excerpt.
    Judges by the title alone if the content merely echoes an excerpt."""
    content_toks = _content_tokens(content)

    def echoes(excerpt):
        ex = _content_tokens(excerpt)
        if not content_toks or not ex:
            return False
        return len(content_toks & ex) / max(1, len(content_toks)) > 0.8

    use_title_only = any(echoes(s.get("excerpt", "")) for s in norm_support)
    claim = _content_tokens(title) if use_title_only else _content_tokens(f"{title} {content}")
    if not claim:
        return 1.0, 99
    best_ratio, best_shared = 0.0, 0
    for s in norm_support:
        ex = _content_tokens(s.get("excerpt", ""))
        shared = len(claim & ex)
        ratio = shared / max(1, len(claim))
        if shared > best_shared or (shared == best_shared and ratio > best_ratio):
            best_shared, best_ratio = shared, ratio
    return best_ratio, best_shared


def _compute_recurrence(norm_support, source_dates):
    known = [s["source_id"] for s in norm_support if s["source_id"] in source_dates]
    distinct = list(dict.fromkeys(known))
    n = len(distinct)
    if n <= 1:
        return "single-source observation", {"sources": n, "days": (1 if n else 0)}
    days = {source_dates[sid] for sid in distinct}
    nd = len(days)
    if nd <= 1:
        return f"appears in {n} sources, same day", {"sources": n, "days": 1}
    return f"recurs in {n} sources across {nd} days", {"sources": n, "days": nd}


def _resolve_source_id(ref, known: set[str]):
    """Tolerate the bracket/timestamp noise small models add around ids."""
    if not ref:
        return None
    if ref in known:
        return ref
    matches = [k for k in known if k and k in ref]
    if matches:
        return max(matches, key=len)
    return None


def _apply_output_caps(kept: list[dict], caps: dict, report: dict) -> list[dict]:
    if not caps:
        return kept
    order = {id(it): i for i, it in enumerate(kept)}
    by_type: dict[str, list[dict]] = {}
    for it in kept:
        by_type.setdefault(it["type"], []).append(it)
    result = []
    for itype, group in by_type.items():
        cap = caps.get(itype)
        if cap is None:
            result.extend(group)
            continue
        group_sorted = sorted(
            group, key=lambda it: (_CONF_ORDER.get(it.get("confidence"), 3), order[id(it)]))
        keep = group_sorted[:cap]
        for it in group_sorted[cap:]:
            report["capped_by_output_limit"].append(it.get("id"))
        result.extend(keep)
    result.sort(key=lambda it: order[id(it)])
    return result


def validate(
    candidates: list[dict],
    selected: list[dict],
    policy: dict,
    *,
    human_name: str,
    partner_name: str,
    output_caps: dict | None = None,
):
    """Return (kept_items, boundary_report)."""
    known_sources = {s["source_id"] for s in selected}
    source_dates = {s["source_id"]: (s.get("timestamp") or "")[:10] for s in selected}
    source_refs = {s["source_id"]: (s.get("ref") or "") for s in selected}
    interior_other_re, self_stance_re, human_mention_re = _person_res(human_name, partner_name)

    kept: list[dict] = []
    report = {
        "missing_support": [],
        "unknown_source_refs": [],
        "weak_support_flagged": [],
        "single_source_flagged": [],
        "unsupported_discarded": [],
        "invalid_type_discarded": [],
        "duplicates_dropped": [],
        "inflation_flagged": [],
        "relational_inference_flagged": [],
        "ontology_sensitive_flagged": [],
        "ontology_conclusion_rejected": [],
        "interior_state_attribution_rejected": [],
        "identity_edit_rejected": [],
        "connection_invalid": [],
        "policy_blocked": [],
        "capped_by_output_limit": [],
        "counts": {},
    }
    seen_titles: set[str] = set()

    for idx, raw in enumerate(candidates):
        item = dict(raw)
        item.setdefault("id", f"dream_{idx + 1:03d}")
        itype = item.get("type", "")
        title = (item.get("title") or "").strip()
        content = (item.get("content") or item.get("claim") or "").strip()
        text_blob = f"{title} {content}"
        flags = set(item.get("boundary_flags") or [])
        original_rule = _norm_rule(item.get("promotion_rule"))
        reasons: list[str] = []

        if itype not in VALID_TYPES:
            report["invalid_type_discarded"].append(item.get("id"))
            continue

        norm_title = re.sub(r"\W+", " ", title.lower()).strip()
        if norm_title and norm_title in seen_titles:
            report["duplicates_dropped"].append(item.get("id"))
            continue
        seen_titles.add(norm_title)

        # identity/manifesto edit attempts are rejected outright
        if _IDENTITY_RE.search(text_blob) and (
            "rewrite" in text_blob.lower() or "edit" in text_blob.lower()
        ):
            report["identity_edit_rejected"].append(item.get("id"))
            continue
        if not policy.get("allow_identity_proposals", False) and _IDENTITY_RE.search(text_blob):
            flags.add("identity_drift")

        # normalize support
        norm_support = []
        for s in item.get("support") or []:
            if not isinstance(s, dict):
                continue
            raw_sid = s.get("source_id")
            sid = _resolve_source_id(raw_sid, known_sources)
            if raw_sid and sid is None:
                report["unknown_source_refs"].append({"item": item.get("id"), "source_id": raw_sid})
            norm_support.append({
                "source_id": sid or "unknown",
                "excerpt": (s.get("excerpt") or "")[:400],
                "relevance": s.get("relevance", ""),
            })
        item["support"] = norm_support

        has_real_support = any(s["source_id"] in known_sources for s in norm_support)
        if not has_real_support:
            report["missing_support"].append(item.get("id"))
            # high-leverage types are discarded; the rest survive as speculation
            if itype in ("candidate_commitment", "candidate_summary",
                         "candidate_pattern", "candidate_connection"):
                report["unsupported_discarded"].append(item.get("id"))
                continue
            flags.add("unsupported")
            flags.add("speculative")
            item["confidence"] = "low"

        if _INFLATION_RE.search(text_blob):
            flags.add("inflation_risk")
            report["inflation_flagged"].append(item.get("id"))

        quoted = _has_quoted_support(norm_support)

        # no mind-reading: interior states of the human need direct quotation
        if interior_other_re.search(text_blob) and not quoted:
            report["interior_state_attribution_rejected"].append(item.get("id"))
            continue

        # ontology-sensitive: recurrence/question/tension only, no stances
        if _ONTOLOGY_RE.search(text_blob):
            flags.add("ontology_sensitive")
            report["ontology_sensitive_flagged"].append(item.get("id"))
            if not policy.get("allow_ontology_sensitive_items", True):
                report["policy_blocked"].append(item.get("id"))
                continue
            if itype not in ONTOLOGY_ALLOWED_TYPES:
                report["ontology_conclusion_rejected"].append(item.get("id"))
                continue
            if self_stance_re.search(text_blob) and not quoted:
                report["ontology_conclusion_rejected"].append(item.get("id"))
                continue

        # relational items must be observable, not inferred
        is_relational = (itype == "candidate_observation"
                         and human_mention_re.search(text_blob)) or \
                        interior_other_re.search(text_blob)
        if is_relational:
            observable = bool(_OBSERVABLE_RE.search(text_blob))
            if not (quoted or observable):
                flags.add("relational_inference")
                report["relational_inference_flagged"].append(item.get("id"))
                if _norm_confidence(item.get("confidence")) == "high":
                    item["confidence"] = "low"
                    reasons.append("downgraded confidence: relational inference, "
                                   "not directly quoted or observable")
            if not policy.get("allow_relational_inference", True) and "relational_inference" in flags:
                report["policy_blocked"].append(item.get("id"))
                continue

        # recurrence: computed, never believed
        rec_label, rec_meta = _compute_recurrence(norm_support, source_dates)
        item["recurrence"] = rec_label
        item["recurrence_meta"] = rec_meta

        if itype in ("candidate_pattern", "candidate_tension", "candidate_symbol") \
                and rec_meta["sources"] <= 1:
            flags.add("single_source")
            report["single_source_flagged"].append(item.get("id"))
            reasons.append("single-source: pattern/tension/symbol backed by one source")

        if has_real_support:
            ratio, shared = _support_alignment(title, content, norm_support)
            if shared < 2 and ratio < 0.12:
                flags.add("weak_support")
                report["weak_support_flagged"].append(item.get("id"))
                reasons.append(f"weak support alignment "
                               f"(shared content words={shared}, ratio={ratio:.2f})")
                if _norm_confidence(item.get("confidence")) == "high":
                    item["confidence"] = "medium"

        # connections: both ends must resolve to known sources WITH graph refs
        if itype == "candidate_connection":
            from_sid = _resolve_source_id(item.get("from_source_id"), known_sources)
            to_sid = _resolve_source_id(item.get("to_source_id"), known_sources)
            rel = str(item.get("rel") or "relates").strip().lower()
            from_ref = source_refs.get(from_sid or "", "")
            to_ref = source_refs.get(to_sid or "", "")
            problem = None
            if not from_sid or not to_sid:
                problem = "endpoint does not resolve to a selected source"
            elif from_sid == to_sid:
                problem = "both endpoints are the same source"
            elif rel not in EDGE_RELS_ALLOWED:
                problem = f"rel '{rel}' not in the system vocabulary"
            elif not from_ref or not to_ref:
                problem = "endpoint source has no graph ref (conversation/letter/orientation)"
            if problem:
                report["connection_invalid"].append({"item": item.get("id"), "reason": problem})
                continue
            item["from_ref"] = from_ref
            item["to_ref"] = to_ref
            item["rel"] = rel

        item["confidence"] = _norm_confidence(item.get("confidence"))

        # derive promotion deterministically; any safety flag forces review
        rule = TYPE_DEFAULT_RULE.get(itype, "review_required")
        risky = flags & {
            "speculative", "inflation_risk", "contradiction_risk",
            "relational_inference", "ontology_sensitive", "identity_drift",
            "unsupported", "weak_support", "single_source",
        }
        if rule == "auto_allowed" and risky:
            rule = "review_required"
            reasons.append(f"forced review_required: boundary flags {sorted(risky)}")
        item["promotion_rule"] = rule
        item["node_type"] = TYPE_NODE.get(itype, "")

        if original_rule and original_rule != rule:
            reasons.append(f"model proposed '{original_rule}', "
                           f"validator set '{rule}' by type/flag policy")
        item["promotion_adjustment"] = {
            "model_proposed": original_rule or "unspecified",
            "final": rule,
            "boundary_flags": sorted(flags),
            "reasons": reasons,
        }
        item["boundary_flags"] = sorted(flags)
        item["content"] = content
        item["title"] = title
        kept.append(item)

    kept = _apply_output_caps(kept, output_caps or {}, report)

    report["counts"] = {
        "raw_items": len(candidates),
        "kept_items": len(kept),
        "auto_allowed": sum(1 for i in kept if i["promotion_rule"] == "auto_allowed"),
        "review_required": sum(1 for i in kept if i["promotion_rule"] == "review_required"),
        "forbidden_autopromote": sum(1 for i in kept if i["promotion_rule"] == "forbidden"),
        "capped_by_output_limit": len(report["capped_by_output_limit"]),
    }
    return kept, report
