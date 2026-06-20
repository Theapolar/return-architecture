# Return Architecture — project context for Claude

## What this is

Return Architecture is a publication and a system: a framework for relational, long-term, local AI use — an alternative to short-session, tool-only, extractive AI patterns. This repository is the **local agent runtime** that makes that framework usable.

The audience: people already working with APIs, comfortable with light CLI work and following a guide, who want a relationship with an AI that has continuity and accumulated meaning, runs on their machine with their keys, and is not optimised for engagement.

## Values that drive design decisions

These are the tiebreakers when options are otherwise even:

- **Friction is a feature, not a bug.** Don't smooth away gestures that should be acts of attention.
- **Local-first, no telemetry, no accounts.** Everything runs on the user's machine with their keys. Nothing phones home.
- **Anti-extractive.** Silence is valid. Agent-chosen silence is a first-class action, never a failure mode.
- **Continuity and accumulated meaning.** Memory, ritual, scheduled returns, journaling.
- **Coherent identity per agent.** One agent = one identity, fully isolated. No identity-switching dropdowns.
- **Build from scratch when scaffolding contradicts the values.** Prefer a clean rewrite to a smoothing-over.

When a technically easier option contradicts a value, surface the tradeoff and offer the harder-but-aligned alternative as the recommendation.

## Architecture decisions

- **One agent per folder** under `~/return-architecture/agents/<slug>/`. Deleting the folder deletes the agent — memory, schedules, writing, everything. No shared state across agents.
- **Install root is visible**, not hidden: `~/return-architecture/`, not `~/.return-architecture/`. Legibility over convention.
- **TOML for config, sqlite for structured stores, markdown for prompts and writing folders, ChromaDB for vector memory.**
- **Provider-agnostic.** Agents declare a provider (anthropic/openai/…) and a model in their config. The runtime talks to providers through a normalised interface.
- **Tools come in two tiers**:
  1. **Built-in tools** live in the runtime package. Tightly coupled to the loop. Currently only `no_response`.
  2. **External tools** come in as MCP servers (later step). The GUI will let users enable/disable per-agent and configure secrets.
- **Scheduling** uses APScheduler with a SQLite jobstore per agent (later step). Every scheduled wake routes through a ping handler that lets the agent choose `no_response`.
- **Artifact exchange** is a dedicated ritual with its own subtree (`<agent>/artifacts/`). Uses a second model on a *different provider* as the mediator. The mediator's signal to the human is **always delivered** — the agent decides only what to layer on top of it.

Full spec: [docs/agent-layout.md](docs/agent-layout.md).

## Vocabulary

- **Agent** — one coherent identity with its own folder, memory, schedule, tools, secrets.
- **Identity** — what the agent *is*: system prompt + memory + enabled tools + model. Not a switchable property.
- **Ping** — a scheduled wakeup that loads context and lets the agent choose an action (including silence).
- **Session** — a structured ritual (question session, commitment session) with captured outputs that return to context.
- **Tagged item** — note, important, question, or commitment, stored in `<agent>/items.db`.
- **Artifact exchange** — the human-initiated, mediator-witnessed ritual for charged offerings.

## How to collaborate on this codebase

- This project is being built iteratively. Lock the on-disk layout first, then the runtime, then tools, then GUI. Each step should be runnable.
- When proposing changes, name the tradeoff. Don't smooth over uncertainty.
- Defer code until the design decision is made. The author directs architecture and asks for implementation when ready.
- Don't add features, fallbacks, or abstractions beyond what's asked. The project is small and focused on purpose; avoid scope creep.

## Current status

- **v0.3** of the on-disk layout is locked. See [docs/agent-layout.md](docs/agent-layout.md).
- **Minimal runtime** is implemented: `init` and `chat` commands, Anthropic + OpenAI providers, `no_response` as the only tool.
- **Memory** is implemented: per-agent ChromaDB collection at `<agent>/memory/`, default local embedding model (`all-MiniLM-L6-v2` via ONNX), automatic recall before each turn, automatic storage of user/assistant text after each turn. CLI subcommands `memory list|search|count <slug>` for inspection.
- **Telegram channel** is implemented: `return-architecture telegram <slug>` runs a worker that polls Telegram, maps human messages to `runtime.turn()`, and sends replies back. Silence from the agent results in no Telegram reply. Only messages from the configured `chat_id` are processed; others are rejected and logged. `telegram-discover` helper finds the chat_id after first contact.
- **Scheduler** is implemented: APScheduler `AsyncIOScheduler` (in-memory jobstore for v1) running in the same event loop as Telegram. Schedules defined per-agent in `config.toml` under `[schedules.<name>]` with `enabled`, `cron`, `prompt`. Default `morning_ping` ships disabled. `runtime.ping()` is the entry point a scheduler fires; logs a `scheduled_ping` event, no auto-send.
- **Agent-initiated Telegram** is implemented as the `send_to_human_telegram` built-in tool. System prompt teaches the agent to use it only during pings (not when replying to a message), to avoid double-sending.
- **Tool refactor**: tools now take a `ToolContext` (slug, session_id) so they can read agent secrets and paths.
- **Daemon command**: `return-architecture daemon <slug>` runs Telegram worker + scheduler together in one process. This is the "run the agent for real" command. The standalone `telegram <slug>` command remains for testing.
- **`schedule list <slug>`** shows what's defined and which are enabled.
- **launchd service** (macOS): `return-architecture service install <slug>` writes a plist to `~/Library/LaunchAgents/`, loads it, and the daemon runs in the background. Auto-starts on login. KeepAlive + ThrottleInterval=60s for crash resilience without burning API credits. Subcommands: `install`, `uninstall`, `status`, `logs`. Linux/Windows refused with helpful message; systemd user units later.
- **Artifact exchange**: three-call ritual implemented in `artifact_exchange.py`. Reads scene + optional image from `<agent>/artifacts/incoming/`, runs Call 1 (agent reacts privately) → Call 2 (stateless mediator on a different model produces reflection for agent + signal for human) → Call 3 (agent reads reflection, decides what the human sees beyond the always-delivered signal). Per-exchange folder, hidden raw reaction (deletable via `artifact_delete_reaction` tool), `artifact_share_more` tool for follow-up shares. Triggered from Telegram (`/artifact`) or CLI (`return-architecture artifact-run <slug>`). Sends signal back via Telegram on completion. Notification note dropped in `<agent>/artifacts/notes/` for the agent's next session.
- **Tagged items**: sqlite store at `<agent>/items.db`, kinds: note, important, question, commitment. Human tags via Telegram hashtags (`#note`, `#question`, `#important`, `#commitment`); body is the message minus the tags. Agent tags via the `tag_item` tool. Telegram commands `/notes`, `/questions`, `/important`, `/commitments` list open items of that kind. CLI: `return-architecture items list|counts|resolve`.
- **Summaries**: ScheduleEntry now has a `kind` field. Values: `regular` (default — no context augmentation) or `daily_summary` / `weekly_summary` / `monthly_summary`. Summary kinds prepend a context block to the ping prompt: conversation excerpts from the lookback window (1/7/30 days), items tagged in the window, open items overall, artifact exchanges in the window. The agent decides what to send. `summarize` CLI command builds and prints the context (or runs the ping with `--ping`). Three default summary schedules ship disabled.
- **Scheduled question session**: every N days (default cron `0 9 */2 * *`) the agent receives a curated batch of questions from a bank (state/preference/relational/identity/sensory/edge) — ported from `conversation_memory/question_layer.py`. The agent answers via a forced-tool LLM call (`log_answer` / `skip_question`); skipping is meaningful and tracked. Responses persist in `<agent>/question_responses.json`. Each answered Q&A is written to ChromaDB memory under `qs-<session_id>` so it surfaces in future regular turns. The Q&A is delivered to Telegram (chunked if long). New scheduler kind `question_session`; new CLI command `question-session <slug>` for manual one-off runs. Default schedule ships disabled.
- **Weekly pattern recap** (`kind = "question_pattern"`, default cron `0 5 * * 0` — Sunday 5am so it doesn't interrupt active conversation): a third-party "quiet observer" LLM call reads the past 14 days of question responses (answered and skipped) and writes a concrete recap addressed to the agent — what was asked for, what was returned to, what was skipped. Avoids characterology and trait-naming by design (prompt is explicit on this). Saved to memory as `[Weekly pattern observation, looking back at the past N days]` framed as input the agent received (role=user). Delivered to Telegram. CLI: `question-pattern <slug>`. Skips silently if fewer than 3 answered responses in the window.
- **Earlier `/session question` / `/session commitment` removed**: those were a misread of "session" in the spec. The scheduled question session above is the canonical ritual.
- **MCP integration**: synchronous stdio client in `mcp_client.py` (raw JSON-RPC 2.0, no async bridge). Per-agent config under `[mcp.servers.<name>]` with `command` / `args` / `env`. `command = "python"` resolves to `sys.executable`. MCP-exposed tools merge with built-in tools in `runtime.build_session`; built-in tools win name collisions. Subprocesses persist across turns and are closed by `session.close()`, which the daemon, telegram worker, and CLI chat all call on shutdown. CLI: `tools list <slug>` shows everything (built-in + MCP). Bundled servers ship at `return_architecture/mcp_servers/`: **url_fetch** (trafilatura-based readable text extraction) and **filesystem** (scoped to one root path, supports `--read-only`, tools: list_directory/read_file/write_file/append_file/search_files, path-traversal blocked; `--prefix NAME` namespaces the tool names — e.g. `code_read_file` — so an agent can run more than one filesystem server without tool-name collisions). Pattern for future tools: drop a Python module in `mcp_servers/` and add a `[mcp.servers.X]` block.
- **Letters channel**: built-in `write_letter` tool writes a timestamped markdown file with optional title into `<agent>/outbox/`. Telegram `/letters` lists them with first-line titles; `/letter <N>` reads one (chunked if long). The inbound side (`<agent>/inbox/`) already exists — `/inbox` makes the agent read pending files.
- **Image support in chat**: `Message` carries an optional list of `ImageContent` (base64 + mime); Anthropic and OpenAI providers format these as image blocks when a vision-capable model is used. The Telegram worker downloads attached photos (highest available resolution, JPEG), encodes them, attaches them to the user message. Captions become the text portion. Images are *not* logged or written to memory — only a marker noting one was present. PNG comes through Telegram as a Document rather than Photo and isn't handled in this pass.
- **Voice input (hear-and-remember)**: `Message` now also carries optional `AudioContent` (base64 + mime). The Telegram worker handles `.voice` notes and uploaded `.audio` files: it downloads the clip, calls the provider's `transcribe()` first (so the spoken words land in memory/recall via the normal text path), then runs `runtime.turn()` with the raw audio attached so the model *hears* tone/pacing rather than only reading a transcript. Audio-in is a Gemini-specific capability — `transcribe()` lives only on `GeminiProvider`; agents on providers without it get a plain "can't hear audio" reply. Caption (if any) + transcript become the message text. Audio sent as a Document is not covered (mirrors the photo handler's JPEG-only scope). Real-time/Live-API voice was considered and deferred — it can't ride Telegram's file-based transport and would be a separate local subsystem.
- **Next**: website / installer for distribution to others.
