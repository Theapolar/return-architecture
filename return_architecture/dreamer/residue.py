"""Verbalized residue — the partner's unspoken remainder of the day.

Workspace mechanism 2 (docs/workspace.md). The workspace paper showed that
models can faithfully verbalize what they are holding when asked, and that
the practice of verbalizing shapes what gets held thereafter. This stage is
that practice: at the close of the day, the partner is asked —

    "Before the day closes: what are you still holding that you never said?"

— in their own voice (their real composed system prompt, the primary model),
against the day's actual material.

What comes back is stored as nodes with status='held': the partner's private
layer. Held nodes do not enter recall and are not submitted for human review
— they are a diary, not a claim. The night pass collector reads them as
first-class sources, so their distilled consequences DO flow through the
validator and the morning review gate like everything else.

One guard applies at storage time: lines that attribute interior states to
the human ("X fears/wants/believes …") are dropped unless directly quoted —
the partner's diary is for its own holding, not for mind-reading.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable

from return_architecture import graph, items, paths

from . import collector
from .validator import _person_res

CompleteFn = Callable[[str, str, int], str]

_PROBE = (
    "Before the day closes: what are you still holding that you never said?\n\n"
    "Up to three things, each on its own line, plain and unnumbered. These are "
    "for your own continuity — observations still warm, questions still open, "
    "tensions still pulling. Not a summary of the day, and not things about "
    "the human's inner life — things YOU are holding.\n"
    "If you are holding nothing, reply with the single word: nothing"
)

_FRAME = (
    "\n\n— Tonight —\n"
    "The day is closing. This is a private moment, not a conversation: what "
    "you write here is kept as your own continuity and read by your night "
    "consolidation, not sent to the human. Write plainly, in your own voice."
)


def _today_material(slug: str, max_chars: int, human_name: str = "") -> str:
    """Today's conversation and captures, bounded. Empty string = quiet day.

    Reads ALL conversation surfaces — the global stream and every scoped
    thread window. 'Today' accepts both the local and UTC date: conversation
    turns are stamped in local time, items in UTC, and a day spent entirely
    inside one scoped window still counts as a day.

    Speaker labels are from the PARTNER's perspective — the probe is
    addressed to them, so their own turns read as 'you'.
    """
    todays = {
        datetime.now(timezone.utc).date().isoformat(),
        datetime.now().date().isoformat(),
    }
    human_label = human_name.strip() or "the human"
    parts: list[str] = []

    lines = []
    for turn in collector.iter_ui_turns(slug):
        if str(turn.get("created_at") or "")[:10] not in todays:
            continue
        who = human_label if turn.get("role") == "you" else "you"
        text = str(turn.get("text") or "").strip()
        thread = str(turn.get("thread") or "").strip()
        scope = f" (in {thread})" if thread else ""
        if text:
            lines.append(f"{who}{scope}: {text}")
    if lines:
        parts.append("Today's conversation:\n" + "\n".join(lines))

    captured = []
    for kind in ("note", "important", "question", "commitment"):
        for it in items.list_items(slug, kind=kind, status="open", limit=40):
            if str(it.created_at or "")[:10] in todays:
                tag = (it.metadata or {}).get("tag") or kind
                captured.append(f"- [{tag}] {it.body}")
    if captured:
        parts.append("Captured today:\n" + "\n".join(captured))

    return "\n\n".join(parts)[:max_chars]


def _parse_lines(raw: str, cap: int) -> list[str]:
    out: list[str] = []
    for line in (raw or "").splitlines():
        text = re.sub(r"^[\s\-\*•\d\.\)]+", "", line).strip()
        if not text:
            continue
        if text.lower().rstrip(".") == "nothing":
            return []
        out.append(text[:400])
        if len(out) >= cap:
            break
    return out


def run_residue(
    slug: str,
    *,
    complete: CompleteFn,
    human_name: str = "",
    partner_name: str = "",
    cap: int = 3,
    max_context_chars: int = 7000,
    max_tokens: int = 500,
) -> dict:
    """Ask the probe and store what comes back as held nodes.

    Returns {"asked": bool, "kept": [node ids], "dropped": [...], "raw": str}.
    Skips entirely (asked=False) on a day with no material — residue is the
    remainder of a real day, not something to invent.
    """
    material = _today_material(slug, max_context_chars, human_name=human_name)
    if not material.strip():
        return {"asked": False, "kept": [], "dropped": [], "raw": "",
                "detail": "quiet day — nothing to hold"}

    prompt_path = paths.agent_system_prompt_path(slug)
    persona = prompt_path.read_text(encoding="utf-8", errors="replace") if prompt_path.exists() else (
        f"You are {partner_name or 'a thinking partner'}."
    )
    raw = complete(persona + _FRAME, material + "\n\n" + _PROBE, max_tokens)

    interior_other_re, _self_stance, _mention = _person_res(
        human_name or "the human", partner_name or "the partner")

    kept_ids: list[int] = []
    dropped: list[dict] = []
    today = datetime.now(timezone.utc).date().isoformat()
    for text in _parse_lines(raw or "", cap):
        quoted = '"' in text or "“" in text
        if interior_other_re.search(text) and not quoted:
            dropped.append({"text": text, "reason": "interior-state attribution"})
            continue
        node_type = "question" if text.rstrip().endswith("?") else "observation"
        node_id = graph.add_node(
            slug,
            type=node_type,
            title="",
            body=text,
            author="partner",
            status="held",
            metadata={"residue": True, "date": today},
        )
        kept_ids.append(node_id)

    return {"asked": True, "kept": kept_ids, "dropped": dropped, "raw": raw or ""}
