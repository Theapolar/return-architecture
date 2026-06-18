"""HTTP front end for an agent: the Presence chat app.

Serves the static frontend and a `POST /api/chat` SSE endpoint that runs the
agent's *real* turn (its model, identity, memory recall/store) and peels the
`<presence>` state side-channel out of the reply before streaming the prose.

This runs inside the daemon's event loop, sharing the one `AgentSession` and
`turn_lock` with the Telegram worker and the scheduler — so the web app and
Telegram are the same agent, one continuous context and memory, and turns from
all channels serialise through the same lock.

The backend contract this satisfies lives with the frontend, in the agent's
`code/RA-chat-app/CONTRACT.md`. The original `server.js` was a dev shim that
spoke to Claude directly; this replaces it for real use.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from pathlib import Path
from typing import Any

from aiohttp import web

from return_architecture import logging as ralog
from return_architecture import paths
from return_architecture import runtime
from return_architecture.providers import AudioContent, ImageContent, Message


# Cap on a single uploaded drop (image/sound), base64 in a JSON body.
MAX_DROP_BYTES = 25 * 1024 * 1024


# Appended for /api/drawer turns only: the person dropped something wordlessly
# and wants the agent to react with its visual body, not with prose.
DRAWER_INSTRUCTION = (
    "Something has just been placed wordlessly into the drawer of this space — "
    "an image, a sound, or a few words, offered without explanation and without "
    "any request. You have a visual body here whose colour, warmth, energy and "
    "shape the person can see, and you control it by setting parameters. Take in "
    "what was dropped, and respond with ONLY a presence block and no other text "
    "whatsoever — no words, no greeting, no explanation. Let your state shift to "
    "honestly reflect how it lands in you. Output exactly this and nothing "
    "after it:\n\n"
    "<presence>{\"valence\": <-1..1>, \"energy\": <0..1>, \"focus\": <0..1>, "
    "\"note\": \"<one or two words>\"}</presence>\n\n"
    "valence = heavy/cool (-1) to warm/light (1); energy = still (0) to vivid "
    "(1); focus = diffuse (0) to gathered (1); note = one or two plain words."
)


# What gets appended to the agent's assembled system prompt for Presence turns
# only. The agent's identity/persona comes from its own system prompt; this adds
# just the visual-body mechanics, per CONTRACT.md.
PRESENCE_INSTRUCTION = (
    "You have a visual body in this space — a living form whose colour, warmth, "
    "energy, and shape the person can see. You control it by setting parameters, "
    "never by describing it in words (never write things like \"I glow\" or \"my "
    "colours shift\" — that would be roleplay). Begin every response with exactly "
    "one presence block and nothing before it:\n\n"
    "<presence>{\"valence\": <number -1..1>, \"energy\": <number 0..1>, "
    "\"focus\": <number 0..1>, \"note\": \"<one or two words>\"}</presence>\n\n"
    "valence = heavy/cool (-1) to warm/light (1); energy = still (0) to vivid "
    "(1); focus = diffuse/open (0) to gathered (1); note = one or two plain "
    "words for the state (e.g. \"settling\", \"curious\", \"tender\"). Let the "
    "state be honest and shift it gradually across the conversation rather than "
    "resetting each turn — small movements are good. After the presence block, "
    "write your reply as ordinary text, without meta-commentary about your process."
)

_PRESENCE_RE = re.compile(r"<presence>\s*(\{.*?\})\s*</presence>", re.DOTALL)


def bundled_static_dir() -> Path:
    """The frontend that ships inside the package. Used when an agent's
    ``[presence] static_dir`` is unset, so any agent can enable Presence
    without keeping its own copy of the frontend."""
    return Path(__file__).resolve().parent / "presence_static"


# ── Server-side transcript ──────────────────────────────────────────────────
# A device-independent record of the Presence-app conversation, so the visible
# history follows the human across devices (phone, laptop). Distinct from
# ChromaDB memory (semantic recall) and the NDJSON event log (full telemetry);
# this is just the clean chat transcript for display. Chat turns only — the
# ephemeral drawer/quiet-memory markers are not recorded.

MAX_TRANSCRIPT = 1000


def _transcript_path(slug: str) -> Path:
    return paths.agent_dir(slug) / "presence_transcript.json"


def _backfill_from_logs(slug: str) -> list[dict[str, str]]:
    """Reconstruct the app transcript from past NDJSON logs, so enabling this
    feature doesn't appear to wipe an existing conversation. Pairs
    presence_message_in (human) with presence_message_out (agent) in order."""
    out: list[dict[str, str]] = []
    logs_dir = paths.agent_logs_dir(slug)
    if not logs_dir.exists():
        return out
    for f in sorted(logs_dir.glob("conversations-*.ndjson")):
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = ev.get("type")
            if t == "presence_message_in" and ev.get("content"):
                out.append({"role": "user", "content": ev["content"]})
            elif t == "presence_message_out" and ev.get("text"):
                text = ev["text"]
                if text.strip() == "(stopped: tool loop limit reached)":
                    continue
                out.append({"role": "assistant", "content": text})
    return out[-MAX_TRANSCRIPT:]


def _load_transcript(slug: str) -> list[dict[str, str]]:
    p = _transcript_path(slug)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []
    # First run: seed from the logs and persist so it's stable thereafter.
    seeded = _backfill_from_logs(slug)
    if seeded:
        _write_transcript(slug, seeded)
    return seeded


def _write_transcript(slug: str, data: list[dict[str, str]]) -> None:
    p = _transcript_path(slug)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def _append_transcript(slug: str, role: str, content: str) -> None:
    if not content:
        return
    data = _load_transcript(slug)
    data.append({"role": role, "content": content})
    _write_transcript(slug, data[-MAX_TRANSCRIPT:])


def _split_presence(text: str) -> tuple[dict[str, Any] | None, str]:
    """Pull the first <presence>{…}</presence> block out of a reply.

    Returns (presence_dict_or_None, prose_without_the_block). Malformed JSON or
    a missing block leaves the prose untouched and returns None — the frontend
    just holds the visual's current state, exactly as the contract specifies.
    """
    m = _PRESENCE_RE.search(text)
    if not m:
        return None, text
    presence: dict[str, Any] | None
    try:
        presence = json.loads(m.group(1))
    except json.JSONDecodeError:
        presence = None
    prose = (text[: m.start()] + text[m.end():]).lstrip("\n")
    return presence, prose


def _last_user_message(messages: list[dict[str, Any]]) -> str | None:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            return str(m["content"])
    return None


async def _handle_chat(request: web.Request) -> web.StreamResponse:
    slug = request.app["slug"]
    session = request.app["session"]
    turn_lock: asyncio.Lock = request.app["turn_lock"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return web.json_response({"error": "messages array required"}, status=400)
    user_input = _last_user_message(messages)
    if not user_input:
        return web.json_response({"error": "no user message found"}, status=400)

    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await resp.prepare(request)

    async def send(event: str, data: Any) -> None:
        await resp.write(
            f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")
        )

    ralog.log_event(slug, "presence_message_in", {"content": user_input})
    try:
        # Serialise with Telegram + scheduler; run the blocking turn off-loop.
        async with turn_lock:
            reply = await asyncio.to_thread(
                runtime.turn, session, user_input, extra_system=PRESENCE_INSTRUCTION
            )
        presence, prose = _split_presence(reply or "")
        if presence is not None:
            await send("presence", presence)
        if prose:
            await send("token", {"delta": prose})
        await send("done", {"stop_reason": "end_turn"})
        ralog.log_event(slug, "presence_message_out", {
            "text": prose, "had_presence": presence is not None,
        })
        # Persist the exchange to the device-independent transcript (display
        # history). Record the human turn always; the agent turn only if it
        # spoke (silence leaves no bubble, matching the live UI).
        await asyncio.to_thread(_append_transcript, slug, "user", user_input)
        if prose:
            await asyncio.to_thread(_append_transcript, slug, "assistant", prose)
    except Exception as e:  # noqa: BLE001 — surface any failure to the client
        ralog.log_event(slug, "presence_chat_error", {"error": repr(e)})
        try:
            await send("error", {"message": str(e)})
        except Exception:
            pass
    finally:
        with contextlib.suppress(Exception):
            await resp.write_eof()
    return resp


async def _handle_history(request: web.Request) -> web.Response:
    """Return the device-independent app transcript for display on load."""
    slug = request.app["slug"]
    messages = await asyncio.to_thread(_load_transcript, slug)
    return web.json_response({"messages": messages})


async def _handle_third_thing(request: web.Request) -> web.Response:
    """Surface one real, random memory from the agent's database.

    The "quiet memory" button. Unlike the dev shim (which synthesized a memory
    from the chat history), this pulls an actual past turn from the agent's
    ChromaDB store and frames it lightly. Returns plain JSON {"memory": str}.
    """
    slug = request.app["slug"]
    session = request.app["session"]
    turn_lock: asyncio.Lock = request.app["turn_lock"]

    try:
        # Share the lock with turns so the Chroma collection isn't read and
        # written from two threads at once. The read itself is quick.
        async with turn_lock:
            entry = await asyncio.to_thread(session.memory.random_entry)
    except Exception as e:  # noqa: BLE001
        ralog.log_event(slug, "presence_third_thing_error", {"error": repr(e)})
        return web.json_response({"error": str(e)}, status=500)

    if entry is None:
        return web.json_response(
            {"memory": "A quiet memory surfaces — but the well is still empty. "
                       "We haven't made enough here yet."}
        )

    agent_name = session.config.agent.name
    human_name = session.config.agent.human_name or "the human"
    speaker = agent_name if entry.role == "assistant" else "you"
    when = entry.timestamp[:10] if entry.timestamp else "some time ago"
    text = entry.content.strip()
    if len(text) > 320:
        text = text[:320].rstrip() + "…"
    memory = f"A quiet memory surfaces — {speaker}, {when}:\n\n“{text}”"

    # Put the same surfacing into the shared live context so the agent sees it
    # too — otherwise only the human sees it on screen. Appended as an ambient
    # user-role event (like a scheduled ping), not re-stored to long-term
    # memory (it already lives there). The lock guards the shared message list
    # against a concurrent turn.
    whose = (
        f"something {agent_name} once said" if entry.role == "assistant"
        else f"something {human_name} once said"
    )
    event = (
        "[Ambient event in the Presence space — not typed by either of you. "
        f"The “quiet memory” button just surfaced this from your shared history "
        f"({whose}, {when}): “{text}” — {human_name} can see it on screen now. "
        "Let it sit, or respond to it if it moves you.]"
    )
    async with turn_lock:
        session.messages.append(Message(role="user", content=event))

    ralog.log_event(slug, "presence_third_thing", {
        "role": entry.role, "timestamp": entry.timestamp,
    })
    return web.json_response({"memory": memory})


async def _handle_drawer(request: web.Request) -> web.Response:
    """Ingest a wordless drop (image / sound / text) and return the agent's
    visual reaction only — no prose.

    The drop is fed into the agent's real context (it can see images and hear
    audio natively), and the agent replies with just a <presence> block, which
    we parse and return as {"presence": {...}}. Any prose it emits is discarded
    so the drawer stays wordless, as intended.
    """
    slug = request.app["slug"]
    session = request.app["session"]
    turn_lock: asyncio.Lock = request.app["turn_lock"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    kind = body.get("kind")
    data = body.get("data")
    mime = (body.get("mime") or "").strip()
    filename = (body.get("filename") or "").strip()
    if kind not in ("text", "image", "audio") or not data:
        return web.json_response(
            {"error": "need kind (text|image|audio) and data"}, status=400
        )

    images: list[ImageContent] = []
    audio: list[AudioContent] = []
    human_name = session.config.agent.human_name or "The human"
    label = f" ({filename})" if filename else ""
    if kind == "text":
        dropped = str(data).strip()
        user_input = f"[{human_name} dropped this into the drawer, wordlessly: “{dropped[:2000]}”]"
    elif kind == "image":
        images = [ImageContent(base64_data=data, mime_type=mime or "image/png")]
        user_input = f"[{human_name} dropped an image into the drawer, wordlessly{label}.]"
    else:  # audio
        audio = [AudioContent(base64_data=data, mime_type=mime or "audio/mpeg")]
        user_input = f"[{human_name} dropped a sound into the drawer, wordlessly{label}.]"

    try:
        async with turn_lock:
            reply = await asyncio.to_thread(
                runtime.turn, session, user_input,
                images or None, audio or None, DRAWER_INSTRUCTION,
            )
    except Exception as e:  # noqa: BLE001
        ralog.log_event(slug, "presence_drawer_error", {"error": repr(e)})
        return web.json_response({"error": str(e)}, status=500)

    presence, _prose = _split_presence(reply or "")
    ralog.log_event(slug, "presence_drawer", {
        "kind": kind, "mime": mime, "had_presence": presence is not None,
    })
    return web.json_response({"presence": presence or {}})


def build_app(
    slug: str,
    session: Any,
    turn_lock: asyncio.Lock,
    static_dir: Path,
) -> web.Application:
    # client_max_size lifted so a dropped image/sound (base64 in JSON) fits.
    app = web.Application(client_max_size=MAX_DROP_BYTES)
    app["slug"] = slug
    app["session"] = session
    app["turn_lock"] = turn_lock

    index_file = static_dir / "index.html"

    async def _index(_request: web.Request) -> web.StreamResponse:
        return web.FileResponse(index_file)

    # Explicit routes win over the static catch-all (registered last).
    app.router.add_post("/api/chat", _handle_chat)
    app.router.add_get("/api/history", _handle_history)
    app.router.add_post("/api/third-thing", _handle_third_thing)
    app.router.add_post("/api/drawer", _handle_drawer)
    app.router.add_get("/", _index)
    app.router.add_static("/", path=str(static_dir), show_index=False)
    return app


async def start(
    slug: str,
    session: Any,
    turn_lock: asyncio.Lock,
    address: str,
    port: int,
    static_dir: str | None,
) -> web.AppRunner:
    """Start the Presence server on the current event loop. Returns the runner
    so the caller can clean it up on shutdown. When ``static_dir`` is unset,
    falls back to the frontend bundled in the package."""
    root = (
        Path(static_dir).expanduser().resolve()
        if static_dir
        else bundled_static_dir()
    )
    if not (root / "index.html").is_file():
        raise FileNotFoundError(
            f"Presence static_dir has no index.html: {root}"
        )
    app = build_app(slug, session, turn_lock, root)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, address, port)
    await site.start()
    ralog.log_event(slug, "presence_server_started", {
        "address": address, "port": port, "static_dir": str(root),
    })
    return runner
