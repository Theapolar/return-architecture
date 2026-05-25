"""Command-line interface."""

from __future__ import annotations

import sys

import typer

import asyncio
import subprocess
import sys
from pathlib import Path

from return_architecture import artifact_exchange as ra_artifact
from return_architecture import daemon as ra_daemon
from return_architecture import init_agent, items as ra_items, memory as ramem, paths, runtime, scheduling
from return_architecture import question_sessions as ra_qs
from return_architecture import service as ra_service
from return_architecture import summaries as ra_summaries
from return_architecture import telegram_worker

app = typer.Typer(
    help="Return Architecture — local agent runtime.",
    no_args_is_help=True,
    add_completion=False,
)

memory_app = typer.Typer(help="Inspect agent memory.", no_args_is_help=True)
app.add_typer(memory_app, name="memory")


@app.command()
def init(
    slug: str = typer.Argument(..., help="Agent slug (lowercase, hyphens)."),
    name: str = typer.Option(None, help="Display name. Defaults to the slug."),
    provider: str = typer.Option("anthropic", help="LLM provider: anthropic | openai."),
    model: str = typer.Option("claude-opus-4-7", help="Model name."),
):
    """Create a new agent and (if missing) the install root."""
    try:
        path = init_agent.create_agent(slug, name=name, provider=provider, model=model)
    except (FileExistsError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Agent created at: {path}")
    typer.echo(f"Install root:     {paths.install_root()}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo(f"  1. Set your API key in: {paths.install_secrets_path()}")
    typer.echo(f"  2. Edit the system prompt at: {paths.agent_system_prompt_path(slug)}")
    typer.echo(f"  3. Start a chat:  return-architecture chat {slug}")


@app.command()
def chat(slug: str = typer.Argument(..., help="Agent slug.")):
    """Interactive REPL with the agent."""
    try:
        session = runtime.build_session(slug)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Talking to '{session.config.agent.name}' "
               f"({session.config.model.provider}/{session.config.model.name})")
    typer.echo("Type ':quit' to exit. Empty lines are ignored.")
    typer.echo("")

    try:
        while True:
            try:
                user_input = input("you: ").strip()
            except (EOFError, KeyboardInterrupt):
                typer.echo("")
                break
            if not user_input:
                continue
            if user_input in (":quit", ":q", ":exit"):
                break

            try:
                reply = runtime.turn(session, user_input)
            except Exception as e:
                typer.echo(f"[error: {e}]", err=True)
                continue

            if reply:
                typer.echo(f"\n{session.config.agent.name}: {reply}\n")
            else:
                typer.echo(f"\n{session.config.agent.name}: (silence)\n")
    finally:
        session.close()


@app.command()
def where():
    """Print the resolved install root and exit."""
    typer.echo(str(paths.install_root()))


@app.command()
def gui(
    port: int = typer.Option(None, help="Port to run on. Defaults to install config or 8501."),
    server_address: str = typer.Option(
        None,
        "--server-address",
        help=(
            "Address to bind the GUI to. Defaults to install config [gui].address, "
            "else 127.0.0.1. Set e.g. to a Tailscale IP to reach the GUI over your "
            "tailnet. The GUI has no auth — never bind 0.0.0.0 without a firewall."
        ),
    ),
):
    """Launch the local Streamlit control panel.

    Reads/writes the same config files the rest of the system uses.
    Open in your browser; Ctrl-C in this terminal to stop.
    """
    from return_architecture.gui import app as gui_app_module

    gui_cfg: dict = {}
    if port is None or server_address is None:
        try:
            install_cfg = paths.install_config_path()
            if install_cfg.exists():
                import tomllib
                with open(install_cfg, "rb") as f:
                    data = tomllib.load(f)
                gui_cfg = data.get("gui") or {}
        except Exception:
            gui_cfg = {}
    if port is None:
        try:
            port = int(gui_cfg.get("port", 8501))
        except (TypeError, ValueError):
            port = 8501
    if server_address is None:
        server_address = str(gui_cfg.get("address", "127.0.0.1"))
    if server_address not in ("127.0.0.1", "localhost"):
        typer.echo(
            f"WARNING: GUI binding to {server_address} — the GUI has no auth; "
            "ensure tailnet/firewall gating.",
            err=True,
        )

    app_file = Path(gui_app_module.__file__)
    _host = "localhost" if server_address in ("127.0.0.1", "localhost") else server_address
    typer.echo(f"Starting GUI at http://{_host}:{port}")
    typer.echo("Ctrl-C to stop.")
    try:
        subprocess.run(
            [
                sys.executable, "-m", "streamlit", "run", str(app_file),
                "--server.port", str(port),
                "--server.address", server_address,
                "--browser.gatherUsageStats", "false",
            ],
        )
    except KeyboardInterrupt:
        pass


@app.command()
def telegram(slug: str = typer.Argument(..., help="Agent slug.")):
    """Run the Telegram channel for an agent. Blocks until Ctrl-C."""
    try:
        telegram_worker.run_worker(slug)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("artifact-run")
def artifact_run(
    slug: str = typer.Argument(..., help="Agent slug."),
    notify_telegram: bool = typer.Option(True, help="Send the result to Telegram if configured."),
):
    """Run an artifact exchange against files in <agent>/artifacts/incoming/.

    Drop a scene.md (or scene.txt) and/or an image into the incoming
    folder, then invoke this command. The three-call ritual runs and
    leaves a per-exchange folder plus a note for the agent's next session.
    """
    try:
        result = ra_artifact.run_exchange(slug)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Exchange complete: {result.exchange_id}")
    typer.echo(f"Folder: {result.exchange_dir}")
    typer.echo(f"Decision: {result.decision}")
    typer.echo("")
    typer.echo("=== for the human ===")
    typer.echo(result.for_human)
    if notify_telegram:
        ra_artifact.notify_human_via_telegram(slug, result)


@app.command("telegram-discover")
def telegram_discover(slug: str = typer.Argument(..., help="Agent slug.")):
    """Find your Telegram chat_id by reading recent updates from the bot."""
    try:
        asyncio.run(telegram_worker.discover_chat_id(slug))
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def daemon(slug: str = typer.Argument(..., help="Agent slug.")):
    """Run the agent for real: Telegram worker + scheduler in one process."""
    try:
        ra_daemon.run_daemon(slug)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


service_app = typer.Typer(
    help="Run the agent daemon as a background launchd service (macOS only).",
    no_args_is_help=True,
)
app.add_typer(service_app, name="service")


@service_app.command("install")
def service_install(slug: str = typer.Argument(..., help="Agent slug.")):
    """Install and start the daemon as a background service.

    The daemon starts immediately, runs in the background, and restarts
    automatically on crash. It also starts on every boot / login.
    """
    import platform
    try:
        config_file = ra_service.install(slug)
    except (RuntimeError, FileNotFoundError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo("Service installed and running.")
    if platform.system() == "Linux":
        typer.echo(f"  unit file: {config_file}")
        typer.echo("")
        typer.echo("Tail logs with:  return-architecture service logs " + slug)
        typer.echo("  (or: journalctl --user -u " + ra_service._linux_unit_name(slug) + " -f)")
    else:
        typer.echo(f"  plist: {config_file}")
        typer.echo(f"  stdout log: {ra_service.stdout_log(slug)}")
        typer.echo(f"  stderr log: {ra_service.stderr_log(slug)}")
        typer.echo("")
        typer.echo("Tail logs with:  return-architecture service logs " + slug)


@service_app.command("uninstall")
def service_uninstall(slug: str = typer.Argument(..., help="Agent slug.")):
    """Stop and remove the background service."""
    try:
        ra_service.uninstall(slug)
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo("Service uninstalled.")


@service_app.command("status")
def service_status(slug: str = typer.Argument(..., help="Agent slug.")):
    """Show whether the service is loaded, its PID, and the config file path."""
    try:
        st = ra_service.status(slug)
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"label:       {st.label}")
    typer.echo(f"config:      {st.config_path}  ({'exists' if st.config_exists else 'missing'})")
    typer.echo(f"loaded:      {'yes' if st.loaded else 'no'}")
    if st.pid is not None:
        typer.echo(f"pid:         {st.pid}")


@service_app.command("logs")
def service_logs(
    slug: str = typer.Argument(..., help="Agent slug."),
    lines: int = typer.Option(40, help="How many lines from each log to show."),
):
    """Show the tail of the service logs."""
    import platform
    try:
        out, err = ra_service.tail_logs(slug, lines=lines)
    except RuntimeError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    if platform.system() == "Linux":
        typer.echo("=== journal ===")
        typer.echo(out if out else "(empty)")
    else:
        typer.echo("=== stdout ===")
        typer.echo(out if out else "(empty)")
        typer.echo("")
        typer.echo("=== stderr ===")
        typer.echo(err if err else "(empty)")


@app.command("question-session")
def question_session_cmd(
    slug: str = typer.Argument(..., help="Agent slug."),
    no_telegram: bool = typer.Option(False, "--no-telegram", help="Skip Telegram delivery; only save responses and memory."),
):
    """Run a question session for an agent now (one-off / for testing).

    Picks a curated batch of questions, asks the agent in a single
    tools-required call (answer or skip per question), saves the
    responses, writes answered Q&A pairs to memory, and (by default)
    sends the Q&A to Telegram.
    """
    try:
        result = ra_qs.run_session(slug, notify_telegram=not no_telegram)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Session {result.session_id}: {result.answered} answered, {result.skipped} skipped.")
    typer.echo("")
    typer.echo(result.message)


@app.command("question-pattern")
def question_pattern_cmd(
    slug: str = typer.Argument(..., help="Agent slug."),
    days: int = typer.Option(14, help="Lookback window in days."),
    min_responses: int = typer.Option(3, help="Minimum answered responses required to run."),
    no_telegram: bool = typer.Option(False, "--no-telegram", help="Skip Telegram delivery."),
):
    """Run an observer pattern-recap over recent question responses."""
    try:
        result = ra_qs.run_pattern_recap(
            slug, days=days, min_responses=min_responses,
            notify_telegram=not no_telegram,
        )
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    if result is None:
        typer.echo(f"Not enough responses in the last {days} days (need at least {min_responses}).")
        return
    typer.echo(f"Pattern recap over {result.days} days: {result.answered_count} answered, {result.skipped_count} skipped.")
    typer.echo("")
    typer.echo(result.text)


@app.command()
def summarize(
    slug: str = typer.Argument(..., help="Agent slug."),
    period: str = typer.Option("daily", help="daily | weekly | monthly"),
    run_ping: bool = typer.Option(
        False,
        "--ping",
        help="Actually invoke the agent with this context (makes API calls). Default: just print the context.",
    ),
):
    """Build a summary context block and optionally fire it through the agent.

    Without --ping, prints the context that would be sent. With --ping,
    runs runtime.ping() so the agent sees the context and may reply
    (e.g. send a Telegram message via send_to_human_telegram).
    """
    kind_map = {"daily": "daily_summary", "weekly": "weekly_summary", "monthly": "monthly_summary"}
    if period not in kind_map:
        typer.echo(f"Error: period must be one of {list(kind_map)}", err=True)
        raise typer.Exit(1)
    kind = kind_map[period]
    lookback = ra_summaries.SUMMARY_KINDS[kind]

    context = ra_summaries.build_summary_context(slug, lookback)
    base_prompt = (
        f"This is a {period} summary moment. Look back at what happened. "
        f"If something stands out, send a brief reflection to the human via "
        f"send_to_human_telegram. If nothing pulls, choose silence."
    )
    full = ra_summaries.render_ping_prompt(base_prompt, context)

    if not run_ping:
        typer.echo("=== ping prompt that would be sent ===\n")
        typer.echo(full)
        return

    try:
        session = runtime.build_session(slug)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Running ping (kind={kind}). This makes LLM calls.")
    result = runtime.ping(session, kind, full)
    if result:
        typer.echo(f"\n=== agent text (not auto-sent) ===\n{result}")
    else:
        typer.echo("\n(agent chose silence or only used tools)")


tools_app = typer.Typer(help="Inspect tools available to an agent.", no_args_is_help=True)
app.add_typer(tools_app, name="tools")


@tools_app.command("list")
def tools_list(slug: str = typer.Argument(..., help="Agent slug.")):
    """Show every tool an agent can call — built-in and MCP-backed."""
    try:
        session = runtime.build_session(slug)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    try:
        for name, tool in sorted(session.tools.items()):
            desc = tool.description.replace("\n", " ")
            if len(desc) > 100:
                desc = desc[:97] + "..."
            typer.echo(f"  {name:35} {desc}")
    finally:
        session.close()


items_app = typer.Typer(help="Inspect tagged items for an agent.", no_args_is_help=True)
app.add_typer(items_app, name="items")


@items_app.command("list")
def items_list(
    slug: str = typer.Argument(..., help="Agent slug."),
    kind: str = typer.Option(None, help=f"Filter by kind: one of {list(ra_items.KINDS)}."),
    status: str = typer.Option("open", help="Status: open, resolved, archived, or 'all' for any."),
    limit: int = typer.Option(50, help="How many to show."),
):
    """Show tagged items for an agent."""
    status_arg = None if status == "all" else status
    rows = ra_items.list_items(slug, kind=kind, status=status_arg, limit=limit)
    if not rows:
        typer.echo("(no items)")
        return
    for it in rows:
        date = it.created_at[:19] if it.created_at else "?"
        typer.echo(f"[{date}] #{it.id} ({it.kind}, {it.source}, {it.status}) {it.body[:200]}")


@items_app.command("counts")
def items_counts(slug: str = typer.Argument(..., help="Agent slug.")):
    """Show open-item counts per kind."""
    counts = ra_items.count_by_kind(slug)
    if not counts:
        typer.echo("(no items)")
        return
    for k in ra_items.KINDS:
        typer.echo(f"  {k:12} {counts.get(k, 0)}")


@items_app.command("resolve")
def items_resolve(
    slug: str = typer.Argument(..., help="Agent slug."),
    item_id: int = typer.Argument(..., help="Item id to mark resolved."),
):
    """Mark an item as resolved."""
    ok = ra_items.resolve_item(slug, item_id)
    typer.echo("resolved" if ok else "(not found or already resolved)")


schedule_app = typer.Typer(help="Inspect agent schedules.", no_args_is_help=True)
app.add_typer(schedule_app, name="schedule")


@schedule_app.command("list")
def schedule_list(slug: str = typer.Argument(..., help="Agent slug.")):
    """Show schedules defined in this agent's config."""
    try:
        handles = scheduling.list_schedules(slug)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    if not handles:
        typer.echo("(no schedules defined)")
        return
    for h in handles:
        mark = "on " if h.enabled else "off"
        typer.echo(f"  [{mark}] {h.name:20}  cron='{h.cron}'")


@memory_app.command("count")
def memory_count(slug: str = typer.Argument(..., help="Agent slug.")):
    """How many entries are in this agent's memory."""
    store = ramem.MemoryStore(slug)
    typer.echo(str(store.count()))


@memory_app.command("list")
def memory_list(
    slug: str = typer.Argument(..., help="Agent slug."),
    limit: int = typer.Option(20, help="How many recent entries to show."),
):
    """Show the most recent memory entries for an agent."""
    store = ramem.MemoryStore(slug)
    entries = store.recent(limit=limit)
    if not entries:
        typer.echo("(no memories yet)")
        return
    for e in entries:
        date = e.timestamp[:19] if e.timestamp else "?"
        typer.echo(f"[{date}] ({e.role}) {e.content[:200]}")


@memory_app.command("search")
def memory_search(
    slug: str = typer.Argument(..., help="Agent slug."),
    query: str = typer.Argument(..., help="What to look for."),
    top_k: int = typer.Option(5, help="How many matches to return."),
):
    """Semantic search through an agent's memory."""
    store = ramem.MemoryStore(slug)
    entries = store.recall(query, top_k=top_k)
    if not entries:
        typer.echo("(no matches)")
        return
    for e in entries:
        date = e.timestamp[:19] if e.timestamp else "?"
        dist = f"{e.distance:.3f}" if e.distance is not None else "?"
        typer.echo(f"[{date}] (d={dist}, {e.role}) {e.content[:200]}")


if __name__ == "__main__":
    app()
