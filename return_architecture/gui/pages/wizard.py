"""First-run setup wizard.

Eight steps that take a fresh install from empty to running:

  0. Welcome
  1. API keys
  2. Create agent
  3. System prompt
  4. Telegram (optional)
  5. Schedules (toggle which to enable)
  6. Install service
  7. Done — opens the dashboard

Triggered when the install has no agents OR no provider keys, and the
session hasn't marked the wizard as done.
"""

from __future__ import annotations

import asyncio

import streamlit as st

from return_architecture import init_agent as ra_init
from return_architecture import service as ra_service
from return_architecture import telegram_worker
from return_architecture.gui import helpers


TOTAL_STEPS = 8


# ── Wizard entry / dispatch ───────────────────────────────────────────────

def start_add_agent() -> None:
    """Re-enter the setup flow to create an *additional* agent.

    Called from the normal GUI (Home page / sidebar) once an install already
    exists. Jumps straight to the create-agent step — the keys are already set
    — and runs the same end-to-end flow (system prompt, Telegram, schedules,
    service) for the new agent. The ``_w_`` prefix means the existing done-step
    cleanup clears the add-mode flag along with the rest of the wizard state.
    """
    for k in list(st.session_state.keys()):
        if k.startswith("_w_"):
            st.session_state.pop(k, None)
    st.session_state.pop("wizard_done", None)
    st.session_state["_w_add_mode"] = True
    st.session_state["wizard_step"] = 2
    st.rerun()


def render() -> None:
    add_mode = st.session_state.get("_w_add_mode")
    st.title("Add another agent" if add_mode else "Welcome to Return Architecture")
    step = st.session_state.get("wizard_step", 0)
    st.progress((step + 1) / TOTAL_STEPS, text=f"Step {step + 1} of {TOTAL_STEPS}")

    if step == 0:
        _step_welcome()
    elif step == 1:
        _step_api_keys()
    elif step == 2:
        _step_create_agent()
    elif step == 3:
        _step_system_prompt()
    elif step == 4:
        _step_telegram()
    elif step == 5:
        _step_schedules()
    elif step == 6:
        _step_install_service()
    else:
        _step_done()


def needs_wizard() -> bool:
    """Should the wizard run instead of the normal navigation?"""
    if st.session_state.get("wizard_done"):
        return False
    if st.session_state.get("wizard_step") is not None:
        return True
    if not helpers.list_agents():
        return True
    secrets = helpers.load_install_secrets_raw()
    if not _any_key_set(secrets):
        return True
    return False


def _any_key_set(secrets: dict) -> bool:
    return (
        helpers.provider_key_set(secrets, "anthropic")
        or helpers.provider_key_set(secrets, "openai")
        or helpers.provider_key_set(secrets, "gemini")
    )


def _advance() -> None:
    st.session_state["wizard_step"] = st.session_state.get("wizard_step", 0) + 1
    st.rerun()


def _go_back() -> None:
    st.session_state["wizard_step"] = max(0, st.session_state.get("wizard_step", 0) - 1)
    st.rerun()


# ── Step 0: Welcome ───────────────────────────────────────────────────────

def _step_welcome() -> None:
    st.markdown(
        """
This is a one-time setup. About 5 minutes. You can change anything later.

What you'll do in the next steps:

1. **Paste an API key** for Anthropic, OpenAI, or Gemini (any one is enough)
2. **Name your agent** and pick a model
3. **Review the agent's system prompt** — its identity in text form
4. **Set up Telegram** so you can message the agent (optional)
5. **Choose scheduled rhythms** — pings, summaries, question sessions (optional)
6. **Install the background service** so the agent runs without a terminal open

When you're done you'll have a working local agent.
"""
    )
    cols = st.columns([5, 1])
    if cols[1].button("Get started →", type="primary", key="_w_start"):
        _advance()


# ── Step 1: API keys ──────────────────────────────────────────────────────

def _step_api_keys() -> None:
    st.subheader("API keys")
    st.write(
        "Paste at least one provider key. You can add the other later. "
        "Keys are stored locally and are never shown back to you after saving."
    )

    secrets = helpers.load_install_secrets_raw()
    anthropic_set = helpers.provider_key_set(secrets, "anthropic")
    openai_set = helpers.provider_key_set(secrets, "openai")
    gemini_set = helpers.provider_key_set(secrets, "gemini")

    st.markdown(f"**Anthropic**: {'✓ set' if anthropic_set else '✗ not set'}")
    new_anth = st.text_input(
        "Anthropic key (paste, or leave empty)",
        type="password",
        key="_w_anth_key",
        placeholder="sk-ant-...",
    )

    st.markdown(f"**OpenAI**: {'✓ set' if openai_set else '✗ not set'}")
    new_oai = st.text_input(
        "OpenAI key (paste, or leave empty)",
        type="password",
        key="_w_oai_key",
        placeholder="sk-...",
    )

    st.markdown(f"**Gemini**: {'✓ set' if gemini_set else '✗ not set'}")
    new_gem = st.text_input(
        "Gemini key (paste, or leave empty) — get one at https://aistudio.google.com/apikey",
        type="password",
        key="_w_gem_key",
        placeholder="AIza...",
    )

    st.divider()
    cols = st.columns([1, 1, 3])
    if cols[0].button("← Back", key="_w_keys_back"):
        _go_back()
    if cols[1].button("Save & continue →", type="primary", key="_w_keys_next"):
        if new_anth.strip():
            helpers.set_provider_key(secrets, "anthropic", new_anth.strip())
        if new_oai.strip():
            helpers.set_provider_key(secrets, "openai", new_oai.strip())
        if new_gem.strip():
            helpers.set_provider_key(secrets, "gemini", new_gem.strip())
        helpers.write_install_secrets(secrets)
        secrets = helpers.load_install_secrets_raw()
        if not _any_key_set(secrets):
            st.error("You need at least one API key to continue.")
        else:
            _advance()


# ── Step 2: Create agent ──────────────────────────────────────────────────

def _step_create_agent() -> None:
    add_mode = st.session_state.get("_w_add_mode")
    st.subheader("Create your agent")
    existing = helpers.list_agents()
    if existing and not add_mode:
        st.info(
            f"You already have agent(s): **{', '.join(existing)}**. "
            "You can skip this step or create another."
        )

    name = st.text_input(
        "Display name",
        value="",
        key="_w_agent_name",
        placeholder="e.g. Aria",
    )
    slug = st.text_input(
        "Slug — the folder name and CLI argument (lowercase letters, digits, hyphens)",
        value="",
        key="_w_agent_slug",
        placeholder="e.g. aria",
    )

    secrets = helpers.load_install_secrets_raw()
    providers_available = []
    if helpers.provider_key_set(secrets, "anthropic"):
        providers_available.append("anthropic")
    if helpers.provider_key_set(secrets, "openai"):
        providers_available.append("openai")
    if helpers.provider_key_set(secrets, "gemini"):
        providers_available.append("gemini")

    if not providers_available:
        st.error("No API keys set — go back to step 1.")
        return

    provider = st.selectbox(
        "Provider",
        options=providers_available,
        key="_w_provider",
    )
    _default_models = {
        "anthropic": "claude-opus-4-8",
        "openai":    "gpt-5.4",
        "gemini":    "gemini-3.1-pro-preview",
    }
    default_model = _default_models.get(provider, "")
    model = st.text_input(
        "Model",
        value=default_model,
        key="_w_model",
        help="e.g. claude-opus-4-8, claude-sonnet-4-6, gpt-5.4, gemini-3.1-pro-preview, gemini-2.5-flash",
    )

    st.divider()
    cols = st.columns([1, 1, 1, 3])
    if cols[0].button("← Back", key="_w_agent_back"):
        _go_back()
    if existing and not add_mode and cols[1].button("Use existing", key="_w_agent_use_existing"):
        st.session_state["_w_created_slug"] = existing[0]
        _advance()
    if cols[2].button("Create →", type="primary", key="_w_agent_create"):
        if not name.strip() or not slug.strip():
            st.error("Both name and slug are required.")
            return
        if slug != slug.lower() or not slug.replace("-", "").replace("_", "").isalnum():
            st.error("Slug must be lowercase letters, digits, and hyphens/underscores.")
            return
        if slug in existing:
            st.error(f"An agent named '{slug}' already exists.")
            return
        try:
            ra_init.create_agent(slug, name=name, provider=provider, model=model)
            st.session_state["_w_created_slug"] = slug
            _advance()
        except (FileExistsError, ValueError) as e:
            st.error(str(e))


# ── Step 3: System prompt ────────────────────────────────────────────────

def _step_system_prompt() -> None:
    slug = st.session_state.get("_w_created_slug")
    if not slug:
        st.error("No agent selected — go back to step 2.")
        return

    st.subheader("System prompt")
    st.write(
        "This is the agent's identity in text form. The default below describes "
        "a coherent agent with continuity, memory, and the option to stay "
        "silent. Edit if you want — or leave it and come back to this on the "
        "System prompt page later."
    )

    current = helpers.load_system_prompt(slug)
    new_value = st.text_area(
        "System prompt",
        value=current,
        height=320,
        key="_w_sysprompt",
        label_visibility="collapsed",
    )

    st.divider()
    cols = st.columns([1, 1, 3])
    if cols[0].button("← Back", key="_w_prompt_back"):
        _go_back()
    if cols[1].button("Save & continue →", type="primary", key="_w_prompt_next"):
        if new_value.strip() != current.strip():
            helpers.write_system_prompt(slug, new_value)
        _advance()


# ── Step 4: Telegram ──────────────────────────────────────────────────────

def _step_telegram() -> None:
    slug = st.session_state.get("_w_created_slug")
    if not slug:
        st.error("No agent selected — go back.")
        return

    st.subheader("Telegram (optional but recommended)")
    st.write(
        "A Telegram bot is how the agent reaches you. If you skip this you "
        "can still chat via CLI: `return-architecture chat <slug>`."
    )

    with st.expander("How to get a bot token", expanded=False):
        st.markdown(
            "1. Open Telegram and search **@BotFather**, then start a chat.\n"
            "2. Send `/newbot`, follow the prompts. Pick any name and a "
            "username ending in `bot`.\n"
            "3. BotFather replies with a **bot token** that looks like "
            "`123456789:ABC-DEF…`. Copy it.\n"
            "4. Send any message to the new bot (e.g. \"hi\") in Telegram.\n"
            "5. Paste the token below, then click **Discover chat ID**."
        )

    secrets = helpers.load_agent_secrets_raw(slug)
    tg = secrets.get("telegram") or {}
    current_token = (tg.get("bot_token") or "").strip()
    current_chat_id = str(tg.get("chat_id") or "").strip()

    st.markdown(f"**Bot token**: {'✓ set' if current_token else '✗ not set'}")
    new_token = st.text_input(
        "Bot token",
        type="password",
        key="_w_tg_token",
        placeholder="123456789:ABC-DEF…",
    )
    if new_token.strip() and new_token.strip() != current_token:
        secrets.setdefault("telegram", {})["bot_token"] = new_token.strip()
        helpers.write_agent_secrets(slug, secrets)
        st.success("Bot token saved.")
        current_token = new_token.strip()

    st.markdown(
        f"**Chat ID**: **{current_chat_id}**"
        if current_chat_id else "**Chat ID**: not set"
    )

    if st.button(
        "Discover chat ID",
        disabled=not current_token,
        key="_w_discover_btn",
    ):
        try:
            with st.spinner("Fetching recent bot updates…"):
                ids = asyncio.run(telegram_worker.fetch_chat_ids(slug))
            st.session_state["_w_discovered"] = ids
        except Exception as e:
            st.error(f"Discover failed: {e}")

    discovered = st.session_state.get("_w_discovered")
    if discovered is not None:
        if not discovered:
            st.info(
                "No recent updates found. Send any message to the bot in "
                "Telegram, then click Discover again."
            )
        else:
            st.write(f"Found {len(discovered)} chat ID(s):")
            for cid, name in discovered:
                if st.button(f"Use {cid}  ({name or '?'})", key=f"_w_use_cid_{cid}"):
                    secrets.setdefault("telegram", {})["chat_id"] = str(cid)
                    helpers.write_agent_secrets(slug, secrets)
                    st.session_state.pop("_w_discovered", None)
                    st.success(f"Chat ID set to {cid}.")
                    st.rerun()

    st.divider()
    cols = st.columns([1, 1, 1, 3])
    if cols[0].button("← Back", key="_w_tg_back"):
        _go_back()
    if cols[1].button("Skip", key="_w_tg_skip"):
        _advance()
    if cols[2].button("Continue →", type="primary", key="_w_tg_next"):
        _advance()


# ── Step 5: Schedules ─────────────────────────────────────────────────────

def _step_schedules() -> None:
    slug = st.session_state.get("_w_created_slug")
    if not slug:
        st.error("No agent selected — go back.")
        return

    st.subheader("Scheduled rhythms")
    st.write(
        "Toggle which scheduled rhythms to enable. All ship disabled by "
        "default — the agent will only react to your messages until you "
        "turn these on. You can adjust schedules anytime on the Schedules page."
    )

    config = helpers.load_agent_config_raw(slug)
    schedules = config.get("schedules") or {}

    new_state: dict[str, bool] = {}
    for name, entry in schedules.items():
        cron = entry.get("cron", "")
        kind = entry.get("kind", "regular")
        is_on = st.checkbox(
            f"**{name}** ({kind}) — `{cron}`",
            value=bool(entry.get("enabled", False)),
            key=f"_w_sched_{name}",
        )
        new_state[name] = is_on

    st.divider()
    cols = st.columns([1, 1, 3])
    if cols[0].button("← Back", key="_w_sched_back"):
        _go_back()
    if cols[1].button("Save & continue →", type="primary", key="_w_sched_next"):
        for name, on in new_state.items():
            config.setdefault("schedules", {}).setdefault(name, {})["enabled"] = on
        helpers.write_agent_config(slug, config)
        _advance()


# ── Step 6: Install service ───────────────────────────────────────────────

def _step_install_service() -> None:
    slug = st.session_state.get("_w_created_slug")
    if not slug:
        st.error("No agent selected — go back.")
        return

    st.subheader("Install the background service")
    st.write(
        "This installs the daemon as a background service "
        "(launchd on macOS, systemd user unit on Linux) and starts it. "
        "The daemon runs Telegram + the scheduler in the background, "
        "survives terminal closure, and auto-starts at login. "
        "You can uninstall anytime from the Service page."
    )

    try:
        status = ra_service.status(slug)
    except RuntimeError as e:
        st.info(str(e))
        st.divider()
        cols = st.columns([1, 1, 3])
        if cols[0].button("← Back", key="_w_svc_back"):
            _go_back()
        if cols[1].button("Skip & continue →", type="primary", key="_w_svc_skip_macos"):
            _advance()
        return

    if status.loaded:
        st.success(f"Service is loaded (PID {status.pid}).")
    else:
        st.info("Service is not loaded yet.")

    if not status.loaded:
        if st.button("Install service", type="primary", key="_w_svc_install"):
            try:
                with st.spinner("Installing…"):
                    ra_service.install(slug)
                st.success("Service installed and running.")
                st.rerun()
            except (RuntimeError, FileNotFoundError) as e:
                st.error(f"Install failed: {e}")

    st.divider()
    cols = st.columns([1, 1, 1, 3])
    if cols[0].button("← Back", key="_w_svc_back2"):
        _go_back()
    if cols[1].button("Skip", key="_w_svc_skip"):
        _advance()
    if status.loaded and cols[2].button("Continue →", type="primary", key="_w_svc_next"):
        _advance()


# ── Step 7: Done ──────────────────────────────────────────────────────────

def _step_done() -> None:
    slug = st.session_state.get("_w_created_slug", "your agent")
    st.subheader("All set 🎉")
    st.markdown(
        f"Your agent **{slug}** is ready.\n\n"
        f"- If you set up Telegram, send the bot a message and the agent will reply.\n"
        f"- You can also chat in a terminal: `return-architecture chat {slug}`.\n"
        f"- Manage everything via the sidebar — system prompt, model, schedules, "
        f"tools, Telegram, letters & items, and service controls."
    )

    if st.button("Open the dashboard →", type="primary", key="_w_done"):
        st.session_state["wizard_done"] = True
        for k in list(st.session_state.keys()):
            if k.startswith("_w_") or k == "wizard_step":
                st.session_state.pop(k, None)
        st.rerun()
