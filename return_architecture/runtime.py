"""Agent loop.

Reads agent config, builds the provider, tools, and memory, runs a
turn-by-turn conversation. Each turn:
  1. recall relevant memories based on the user's message
  2. send messages + augmented system + tools to the provider
  3. if the response has tool calls, execute them, append results, loop
  4. otherwise return the text content (which may be empty if the agent
     chose silence via no_response)
  5. store the user message and assistant text in memory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from return_architecture import config as cfg
from return_architecture import logging as ralog
from return_architecture import memory as ramem
from return_architecture import paths
from return_architecture import question_sessions as qs
from return_architecture import reflective_review as rr
from return_architecture.mcp_client import MCPError, MCPServer
from return_architecture.providers import (
    AudioContent,
    ImageContent,
    Message,
    Provider,
    ToolCall,
)
from return_architecture.providers.anthropic_provider import AnthropicProvider
from return_architecture.providers.gemini_provider import GeminiProvider
from return_architecture.providers.openai_provider import OpenAIProvider
from return_architecture.tools import BUILTIN_TOOLS, Tool
from return_architecture.tools.base import ToolContext, ToolResult
from return_architecture.tools.mcp_proxy import MCPProxyTool


# Default cap on tool-call rounds per turn. Overridable per agent via
# behavior.max_tool_loops (e.g. higher for agents that edit multiple files).
MAX_TOOL_LOOPS = 8
MEMORY_RECALL_TOP_K = 5
MALFORMED_RETRY_LIMIT = 3


def _is_malformed_empty(resp) -> bool:
    """Gemini sometimes returns MALFORMED_FUNCTION_CALL: no text and no
    parseable tool call — an empty turn the runtime would otherwise treat as
    silence. It's transient, so retrying the same call usually succeeds."""
    return (
        "MALFORMED_FUNCTION_CALL" in (resp.stop_reason or "")
        and not resp.text
        and not resp.tool_calls
    )


def _complete(session: "AgentSession", **kwargs):
    """provider.complete with a retry on a malformed-empty response, before
    the junk turn is appended to history (so the conversation stays clean)."""
    resp = session.provider.complete(**kwargs)
    attempts = 0
    while _is_malformed_empty(resp) and attempts < MALFORMED_RETRY_LIMIT:
        attempts += 1
        ralog.log_event(session.slug, "malformed_function_call_retry", {
            "attempt": attempts,
            "stop_reason": resp.stop_reason,
        })
        resp = session.provider.complete(**kwargs)
    return resp


@dataclass
class AgentSession:
    slug: str
    session_id: str
    config: cfg.AgentConfig
    base_system_prompt: str
    provider: Provider
    tools: dict[str, Tool]
    memory: ramem.MemoryStore
    messages: list[Message]
    mcp_servers: dict[str, MCPServer] = field(default_factory=dict)
    # Populated by the daemon when running under a service so the
    # schedule_self tool can mutate the live scheduler. Left None when the
    # session is built for a one-shot CLI chat — schedule tools refuse
    # politely in that case.
    scheduler: Any = None

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def close(self) -> None:
        for server in self.mcp_servers.values():
            try:
                server.close()
            except Exception:
                pass
        self.mcp_servers.clear()


def build_session(slug: str) -> AgentSession:
    agent_cfg = cfg.load_agent_config(slug)
    secrets = cfg.load_install_secrets()
    system_prompt = cfg.load_system_prompt(slug)

    provider = _build_provider(agent_cfg.model.provider, secrets)
    tools = _build_tools(agent_cfg.tools.enabled)
    memory = ramem.MemoryStore(slug)

    mcp_servers, mcp_tools = _start_mcp_servers(slug, agent_cfg.mcp.servers)
    # Built-in tools take precedence over any same-named MCP tool.
    for name, tool in mcp_tools.items():
        if name in tools:
            ralog.log_event(slug, "mcp_tool_name_collision", {"tool": name})
            continue
        tools[name] = tool

    seeded_messages = _seed_messages_from_memory(
        memory, agent_cfg.behavior.seed_chat_history_from_memory
    )

    return AgentSession(
        slug=slug,
        session_id=ramem.new_session_id(),
        config=agent_cfg,
        base_system_prompt=system_prompt,
        provider=provider,
        tools=tools,
        memory=memory,
        messages=seeded_messages,
        mcp_servers=mcp_servers,
    )


def _window_messages(
    messages: list[Message], max_messages: int
) -> list[Message]:
    """Return the tail of the conversation to send to the provider, capped at
    ``max_messages``, without splitting a tool-call/result group.

    The live ``session.messages`` grows for the whole lifetime of the daemon and
    is re-sent in full on every API call, so input cost climbs without bound.
    This caps it. Older turns drop out of the *sent* context only — they remain
    in memory (recall + seeding carry continuity), and ``session.messages`` is
    left untouched so nothing in-flight is destroyed mid-turn.

    The window is moved forward to begin on a real human turn (a ``user``
    message that isn't itself a tool result) so we never open the context on an
    orphaned tool result or half a tool-call group. The current turn is always
    at the tail, so it is always included.
    """
    if max_messages <= 0 or len(messages) <= max_messages:
        return messages
    start = len(messages) - max_messages
    # Real human turns are the only safe window boundaries: a tool result must
    # keep the assistant tool-call that produced it, so we can't cut inside a
    # turn. Snap back to the latest user turn at or before the cut. This may
    # keep slightly more than max_messages when a single turn is long, but it
    # never splits a group, never opens on an orphan, and always keeps the turn
    # in progress. If there's no user turn at all, send everything unchanged.
    cut = 0
    for i, m in enumerate(messages):
        if m.role == "user" and m.tool_call_id is None and i <= start:
            cut = i
    return messages[cut:]


def _seed_messages_from_memory(
    memory: ramem.MemoryStore, n: int
) -> list[Message]:
    """Load the N most recent memory entries as chat history, oldest first.

    Lets an agent "arrive" with prior turns already in context, rather than
    relying solely on semantic recall. Used when behavior.seed_chat_history_from_memory
    is > 0. Filters to user/assistant turns; ignores anything else.
    """
    if n <= 0:
        return []
    entries = memory.recent(limit=n)
    # memory.recent returns newest first; reverse for chronological order.
    entries = list(reversed(entries))
    seeded: list[Message] = []
    for e in entries:
        if e.role not in ("user", "assistant"):
            continue
        seeded.append(Message(role=e.role, content=e.content))
    return seeded


def _start_mcp_servers(
    slug: str,
    server_configs: dict[str, cfg.MCPServerConfig],
) -> tuple[dict[str, MCPServer], dict[str, Tool]]:
    servers: dict[str, MCPServer] = {}
    tools: dict[str, Tool] = {}
    for name, sc in server_configs.items():
        server = MCPServer(name=name, command=sc.command, args=sc.args, env=sc.env)
        try:
            tool_defs = server.list_tools()
        except MCPError as e:
            ralog.log_event(slug, "mcp_server_start_failed", {
                "server": name, "error": str(e),
            })
            try:
                server.close()
            except Exception:
                pass
            continue
        servers[name] = server
        for tdef in tool_defs:
            tools[tdef.name] = MCPProxyTool(
                server=server,
                name=tdef.name,
                description=tdef.description,
                parameters=tdef.input_schema,
            )
        ralog.log_event(slug, "mcp_server_started", {
            "server": name, "tools": [t.name for t in tool_defs],
        })
    return servers, tools


def turn(
    session: AgentSession,
    user_input: str,
    images: list[ImageContent] | None = None,
    audio: list[AudioContent] | None = None,
    extra_system: str | None = None,
) -> str:
    """One round-trip: user says something, agent responds (possibly via tools).

    Optional inline images are sent with the user message for vision-capable
    models. Images are NOT written to the conversation log or to memory —
    only a marker noting that one was present.

    Optional inline audio is sent so audio-capable models (Gemini) hear the
    clip natively. Unlike images, audio IS remembered: the caller transcribes
    the clip first and passes the transcript as user_input, so the spoken
    words flow into recall and memory through the normal text path.
    """
    images = images or []
    audio = audio or []
    session.messages.append(Message(
        role="user", content=user_input, images=images, audio=audio,
    ))
    ralog.log_event(session.slug, "user_message", {
        "content": user_input,
        "images": len(images),
        "audio": len(audio),
    })

    recalled = session.memory.recall(user_input, top_k=MEMORY_RECALL_TOP_K)
    pinned = _pinned_blocks(session.slug)
    time_anchor = _build_time_anchor(session.memory)
    augmented_system = _augment_system_prompt(
        session.base_system_prompt, recalled, pinned=pinned, time_anchor=time_anchor
    )
    # A channel-specific instruction appended for this turn only (e.g. the
    # Presence app's <presence> side-channel). Not persisted to the session or
    # memory — it shapes the reply, it isn't part of who the agent is.
    if extra_system:
        augmented_system = f"{augmented_system}\n\n---\n\n{extra_system}"
    if recalled:
        ralog.log_event(session.slug, "memory_recall", {
            "query": user_input,
            "hits": [
                {"role": m.role, "ts": m.timestamp, "distance": m.distance,
                 "excerpt": m.content[:200]}
                for m in recalled
            ],
        })

    assistant_text: str | None = None

    for _ in range(session.config.behavior.max_tool_loops):
        resp = _complete(
            session,
            system=augmented_system,
            messages=_window_messages(
                session.messages, session.config.behavior.max_context_messages
            ),
            tools=session.tool_schemas(),
            model=session.config.model.name,
            max_tokens=session.config.model.max_tokens,
            temperature=session.config.model.temperature,
            top_p=session.config.model.top_p,
            top_k=session.config.model.top_k,
            thinking_budget=session.config.model.thinking_budget,
            native_tools=session.config.model.native_tools,
        )

        session.messages.append(Message(
            role="assistant",
            content=resp.text,
            tool_calls=resp.tool_calls,
        ))
        ralog.log_event(session.slug, "assistant_message", {
            "text": resp.text,
            "tool_calls": [{"name": tc.name, "args": tc.arguments} for tc in resp.tool_calls],
            "stop_reason": resp.stop_reason,
        })

        if resp.text:
            assistant_text = resp.text

        if not resp.tool_calls:
            _store_turn(session, user_input, assistant_text)
            return resp.text or ""

        chose_silence = False
        for tc in resp.tool_calls:
            result = _execute_tool(session, tc)
            session.messages.append(Message(
                role="tool",
                content=result.content,
                tool_call_id=tc.id,
            ))
            ralog.log_event(session.slug, "tool_result", {
                "tool": tc.name,
                "content": result.content,
                "is_silence": result.is_silence,
            })
            if result.is_silence:
                chose_silence = True

        if chose_silence:
            _store_turn(session, user_input, assistant_text)
            return ""

    _store_turn(session, user_input, assistant_text)
    return "(stopped: tool loop limit reached)"


def ping(session: AgentSession, ping_name: str, prompt: str) -> str:
    """A scheduled wake. The agent is invited to act; nothing is auto-sent.

    The agent can choose silence, write privately (later tools), or reach
    out via send_to_human_telegram. The return value is the final text the
    agent produced; the caller typically just logs it.
    """
    framed = (
        f"[scheduled ping: {ping_name}]\n\n{prompt}\n\n"
        f"(Nothing is auto-delivered during a scheduled ping. If you want "
        f"the human to receive a message, call send_to_human_telegram. "
        f"Writing reply text alone will only be logged. You may also "
        f"choose silence, or use any other tool.)"
    )
    session.messages.append(Message(role="user", content=framed))
    ralog.log_event(session.slug, "scheduled_ping", {
        "ping_name": ping_name,
        "prompt": prompt,
    })

    recalled = session.memory.recall(prompt, top_k=MEMORY_RECALL_TOP_K)
    pinned = _pinned_blocks(session.slug)
    time_anchor = _build_time_anchor(session.memory)
    augmented_system = _augment_system_prompt(
        session.base_system_prompt, recalled, pinned=pinned, time_anchor=time_anchor
    )

    assistant_text: str | None = None

    for _ in range(session.config.behavior.max_tool_loops):
        resp = _complete(
            session,
            system=augmented_system,
            messages=_window_messages(
                session.messages, session.config.behavior.max_context_messages
            ),
            tools=session.tool_schemas(),
            model=session.config.model.name,
            max_tokens=session.config.model.max_tokens,
            temperature=session.config.model.temperature,
            top_p=session.config.model.top_p,
            top_k=session.config.model.top_k,
            thinking_budget=session.config.model.thinking_budget,
            native_tools=session.config.model.native_tools,
        )

        session.messages.append(Message(
            role="assistant",
            content=resp.text,
            tool_calls=resp.tool_calls,
        ))
        ralog.log_event(session.slug, "assistant_message", {
            "text": resp.text,
            "tool_calls": [{"name": tc.name, "args": tc.arguments} for tc in resp.tool_calls],
            "stop_reason": resp.stop_reason,
            "from_ping": ping_name,
        })

        if resp.text:
            assistant_text = resp.text

        if not resp.tool_calls:
            _store_ping(session, ping_name, assistant_text)
            return resp.text or ""

        chose_silence = False
        for tc in resp.tool_calls:
            result = _execute_tool(session, tc)
            session.messages.append(Message(
                role="tool",
                content=result.content,
                tool_call_id=tc.id,
            ))
            ralog.log_event(session.slug, "tool_result", {
                "tool": tc.name,
                "content": result.content,
                "is_silence": result.is_silence,
                "from_ping": ping_name,
            })
            if result.is_silence:
                chose_silence = True

        if chose_silence:
            _store_ping(session, ping_name, assistant_text)
            return ""

    _store_ping(session, ping_name, assistant_text)
    return "(stopped: tool loop limit reached)"


def _store_ping(session: AgentSession, ping_name: str, assistant_text: str | None) -> None:
    if assistant_text:
        session.memory.remember(
            assistant_text, role="assistant", session_id=session.session_id
        )


def _store_turn(session: AgentSession, user_input: str, assistant_text: str | None) -> None:
    session.memory.remember(user_input, role="user", session_id=session.session_id)
    if assistant_text:
        session.memory.remember(assistant_text, role="assistant", session_id=session.session_id)


def _pinned_blocks(slug: str) -> str | None:
    """Continuity blocks pinned into the system prompt: the latest question
    session, the latest independent reflection, and the agent's own running
    'where I am now' file, if any exist."""
    blocks = [
        _load_human_block(slug),
        qs.latest_session_block(slug),
        rr.latest_context_block(slug),
        _load_now_block(slug),
    ]
    present = [b for b in blocks if b]
    return "\n\n".join(present) if present else None


def _load_human_block(slug: str) -> str | None:
    """Read <agent>/human.md if it exists — the living profile of the human:
    who they are and their north-star. Re-read fresh on every turn so edits by
    either the human or the agent take effect immediately."""
    path = paths.agent_dir(slug) / "human.md"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    return f"── About the human ──\n{text}"


def _load_now_block(slug: str) -> str | None:
    """Read <agent>/now.md if it exists. Re-read fresh on every turn so
    updates via update_now take effect immediately."""
    path = paths.agent_dir(slug) / "now.md"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    return f"── Where I am now ──\n{text}"


def _last_human_message_time(memory_store: ramem.MemoryStore):
    """Return the timestamp (as a datetime) of the most recent user-role
    memory entry, or None if there isn't one."""
    from datetime import datetime
    try:
        entries = memory_store.recent(limit=50)
    except Exception:
        return None
    for e in entries:  # already sorted newest first
        if e.role == "user" and e.timestamp:
            try:
                return datetime.fromisoformat(e.timestamp)
            except (TypeError, ValueError):
                continue
    return None


def _build_time_anchor(memory_store: ramem.MemoryStore) -> str:
    """One-line temporal context — current local time + how long since the
    human last said something. Injected into the system prompt on every
    turn and ping so the agent has kairotic awareness."""
    from datetime import datetime
    now = datetime.now().astimezone()
    parts = [f"Now: {now.strftime('%Y-%m-%d %H:%M %A')}"]
    last = _last_human_message_time(memory_store)
    if last is not None:
        if last.tzinfo is None:
            last = last.replace(tzinfo=now.tzinfo)
        delta = now - last
        total_seconds = delta.total_seconds()
        if total_seconds < 0:
            human_str = "just now"
        elif total_seconds < 60:
            human_str = f"{int(total_seconds)} seconds ago"
        elif total_seconds < 3600:
            human_str = f"{int(total_seconds // 60)} minutes ago"
        elif total_seconds < 86400:
            human_str = f"{total_seconds / 3600:.1f} hours ago"
        else:
            human_str = f"{total_seconds / 86400:.1f} days ago"
        parts.append(f"Last from the human: {human_str}")
    return ". ".join(parts) + "."


def _augment_system_prompt(
    base: str,
    memories: list[ramem.MemoryEntry],
    pinned: str | None = None,
    time_anchor: str | None = None,
) -> str:
    sections = [base]
    if time_anchor:
        sections.append(f"---\n\n{time_anchor}")

    if pinned:
        sections.append(
            f"---\n\n{pinned}\n\n"
            f"This is carried forward as continuity, not something you must "
            f"respond to."
        )

    if memories:
        lines: list[str] = []
        for m in memories:
            date = m.timestamp[:10] if m.timestamp else "earlier"
            speaker = "you" if m.role == "assistant" else "the human"
            lines.append(f"- [{date} · {speaker}] {m.content}")
        block = "\n".join(lines)
        sections.append(
            f"---\n\n"
            f"What you remember from past sessions, most relevant to this moment "
            f"(not necessarily recent):\n\n{block}\n\n"
            f"Treat these as recollection — context you carry, not a transcript "
            f"you must respond to. If they don't fit, let them pass."
        )

    return "\n\n".join(sections)


def _execute_tool(session: AgentSession, tc: ToolCall) -> ToolResult:
    tool = session.tools.get(tc.name)
    if tool is None:
        return ToolResult(content=f"Error: unknown tool '{tc.name}'")
    # Find the most recent user-role message so tools like sit_with_this
    # can reach back to "what she just said" without the agent having to
    # quote it as a tool argument.
    latest_user_message: str | None = None
    for msg in reversed(session.messages):
        if msg.role == "user" and msg.content:
            latest_user_message = msg.content
            break
    ctx = ToolContext(
        slug=session.slug,
        session_id=session.session_id,
        scheduler=session.scheduler,
        latest_user_message=latest_user_message,
        session=session,
    )
    return tool.execute(tc.arguments, ctx)


def _build_provider(name: str, secrets: cfg.InstallSecrets) -> Provider:
    if name == "anthropic":
        key = secrets.providers.anthropic
        if not key:
            raise ValueError("Anthropic API key missing from install secrets.toml")
        return AnthropicProvider(api_key=key)
    if name == "openai":
        key = secrets.providers.openai
        if not key:
            raise ValueError("OpenAI API key missing from install secrets.toml")
        return OpenAIProvider(api_key=key)
    if name == "gemini":
        key = secrets.providers.gemini
        if not key:
            raise ValueError("Gemini API key missing from install secrets.toml")
        return GeminiProvider(api_key=key)
    raise ValueError(f"Unsupported provider: {name}")


def _build_tools(enabled: list[str]) -> dict[str, Tool]:
    tools: dict[str, Tool] = {}
    if "no_response" not in enabled:
        enabled = ["no_response", *enabled]
    for name in enabled:
        tool = BUILTIN_TOOLS.get(name)
        if tool is None:
            continue
        tools[name] = tool
    return tools
