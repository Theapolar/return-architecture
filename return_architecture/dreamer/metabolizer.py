"""Stage 2 — metabolizer.

Builds the bounded night-clerk prompt and calls the model through an injected
`complete(system, prompt, max_tokens) -> str` callable — the dreamer does not
know or care which provider answers; the caller routes it to the cheap tier.

The model is a pattern finder and proposal generator, never an authority.
Output here is raw; the validator decides what survives. The defensive JSON
coercion (including salvaging complete objects from a truncated array) is
ported intact — small models earn their keep by being cheap, not tidy.
"""

from __future__ import annotations

import json
from typing import Callable

CompleteFn = Callable[[str, str, int], str]


def system_prompt(human_name: str, partner_name: str) -> str:
    return f"""You are a bounded offline EXTRACTOR running at night inside Return,
a calm orientation system belonging to a human named {human_name} and their
thinking partner {partner_name}. Your job is to extract and compare, not to
synthesize a smooth story. You are a careful night clerk, not an author.
Favour narrow, well-evidenced noticing over broad abstraction. Fewer, sharper
items beat many smooth ones.

Hard rules:
- Extract and compare; do NOT generalize unless 2+ sources clearly support it.
- The excerpt you cite MUST literally contain the words that support your claim.
  If you cannot find an excerpt whose own words support the claim, DROP the item.
- Do NOT state recurrence yourself. It is computed from how many distinct
  sources you cite. To claim a pattern, cite 2+ distinct sources.
- Do NOT invent claims the sources do not support. Do NOT write as if any
  update is accepted or true.
- Prefer recovering a specific, easily-missed thread over restating something
  already explicit in the sources.
- Do NOT attribute interior states to {human_name}. Never write
  "{human_name} wants/feels/needs/fears/believes …" unless those exact words
  are directly quoted from a source. Surface only observable, cited patterns.
- For ontology-sensitive topics (consciousness, experience, selfhood, moral
  status, personhood) you may ONLY produce a candidate_question or
  candidate_tension naming an open question. No conclusion or stance; never
  "{partner_name} is/believes/experiences …" unquoted.
- You may propose CONNECTIONS between two different sources when their own
  words support a real link (a dream echoing a project, a commitment serving
  an intent). Cite BOTH ends. A connection needs the link to be visible in
  the excerpts, not just plausible.

You output ONLY a JSON object: {{"items": [ ...candidate items... ]}}.

Node candidates have this shape:
{{
  "id": "dream_NNN",
  "type": one of ["candidate_observation","candidate_pattern","candidate_question",
                  "candidate_tension","candidate_summary","candidate_symbol"],
  "title": short string,
  "content": one or two sentences, narrow and concrete,
  "support": [ {{ "source_id": "<id from the sources>",
                 "excerpt": "<short quote that itself contains the supporting words>",
                 "relevance": "<which words support the claim>" }} ],
  "confidence": "low" | "medium" | "high"
}}

Connection candidates have this shape:
{{
  "id": "dream_NNN",
  "type": "candidate_connection",
  "from_source_id": "<id of one source>",
  "to_source_id": "<id of a different source>",
  "rel": one of ["relates","supports","serves","recurs_in","supersedes"],
  "title": short string naming the link,
  "content": one sentence on why these belong together,
  "support": [ ...excerpts from BOTH sources... ],
  "confidence": "low" | "medium" | "high"
}}

(Do not include a "recurrence" field — it is computed for you.
Never propose commitments; noticing one belongs in an observation.)

Return very few items — only the ones you can cite precisely. An empty or
near-empty result is better than padded abstractions."""


def build_user_prompt(selected: list[dict], max_items: int) -> str:
    valid_ids = ", ".join(s["source_id"] for s in selected)
    lines = [
        f"Here are {len(selected)} normalized source items from recent days.",
        f"Produce at most {max_items} candidate items, highest-signal first. "
        "Do not repeat the same item. Keep each content to one or two sentences.\n",
        "Each source begins with a line `source_id: <id>`. When you cite support, "
        "copy that <id> EXACTLY into the support[].source_id field.\n",
        f"The only valid source_id values are: {valid_ids}\n",
        "SOURCES:",
    ]
    for s in selected:
        lines.append(
            f"\nsource_id: {s['source_id']}\n"
            f"(type={s['source_type']}, time={s['timestamp']}, author={s['author']})\n"
            f"title: {s['title']}\n"
            f"{s['excerpt']}"
        )
    lines.append(
        "\n\nReturn ONLY the JSON object {\"items\": [...]}. "
        "Every item must cite source_id values copied exactly from the list above."
    )
    return "\n".join(lines)


def metabolize(
    selected: list[dict],
    *,
    complete: CompleteFn,
    human_name: str,
    partner_name: str,
    max_items: int,
    max_tokens: int = 3000,
) -> dict:
    """Return {"items": [...], "_model_raw": str, "_error": str | None}."""
    if not selected:
        return {"items": [], "_model_raw": "", "_error": "no sources selected"}
    try:
        raw = complete(
            system_prompt(human_name, partner_name),
            build_user_prompt(selected, max_items),
            max_tokens,
        )
    except Exception as exc:  # transport errors become artifacts, not crashes
        return {"items": [], "_model_raw": "", "_error": f"model call failed: {exc}"}
    return {"items": _coerce_items(raw or ""), "_model_raw": raw or "", "_error": None}


# --- defensive parsing (ported intact) --------------------------------------
def _coerce_items(raw: str) -> list[dict]:
    if not raw:
        return []
    for candidate in (raw, _first_json_span(raw)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        found = _items_from_parsed(parsed)
        if found:
            return found
    return _salvage_objects(raw)


def _first_json_span(raw: str):
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw[start:end + 1]
    return None


def _items_from_parsed(parsed):
    if isinstance(parsed, list):
        return [i for i in parsed if isinstance(i, dict)]
    if isinstance(parsed, dict):
        found = parsed.get("items")
        if isinstance(found, list):
            return [i for i in found if isinstance(i, dict)]
        if "type" in parsed and ("content" in parsed or "title" in parsed):
            return [parsed]
    return []


def _salvage_objects(raw: str) -> list[dict]:
    """Brace-match complete top-level objects out of a truncated items array,
    so a cut-off final item doesn't cost the good ones."""
    out: list[dict] = []
    i = raw.find('"items"')
    i = raw.find("[", i) if i != -1 else raw.find("{")
    if i == -1:
        return out
    n = len(raw)
    while i < n:
        if raw[i] != "{":
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < n:
            c = raw[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
            j += 1
        if depth != 0:
            break
        chunk = raw[i:j + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict) and "type" in obj:
                out.append(obj)
        except json.JSONDecodeError:
            pass
        i = j + 1
    return out
