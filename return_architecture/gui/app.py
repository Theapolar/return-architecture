"""Streamlit entry point for the Return Architecture control panel.

Invoked via the CLI: `return-architecture gui`.

If the install hasn't been set up yet (no agents, or no API keys), this
routes to the first-run setup wizard instead of the normal navigation.
"""

from __future__ import annotations

import streamlit as st

from return_architecture.gui import helpers
from return_architecture.gui.pages import (
    browse,
    home,
    identity,
    install_settings,
    overview,
    schedules,
    service,
    system_prompt,
    telegram,
    tools,
    wizard,
)


def _main() -> None:
    st.set_page_config(
        page_title="Return Architecture",
        layout="wide",
    )

    if wizard.needs_wizard():
        wizard.render()
        return

    agents = helpers.list_agents()
    install_cfg = helpers.load_install_config_raw()

    # Agent selector at the top of the sidebar (above the page list).
    if agents:
        default_agent = (install_cfg.get("install") or {}).get("default_agent") or agents[0]
        if default_agent not in agents:
            default_agent = agents[0]

        current = st.session_state.get("agent_slug") or default_agent
        if current not in agents:
            current = default_agent

        with st.sidebar:
            selected = st.selectbox(
                "Agent",
                options=agents,
                index=agents.index(current),
                key="_agent_select",
            )
            st.session_state["agent_slug"] = selected
            if st.button("➕ Add another agent", key="_sidebar_add_agent", use_container_width=True):
                wizard.start_add_agent()
            st.divider()

    pages = [
        st.Page(home.render, title="Home", icon="🏠", url_path="home", default=True),
    ]
    if agents:
        pages.append(
            st.Page(overview.render, title="Overview", icon="📊", url_path="overview")
        )
        pages.append(
            st.Page(identity.render, title="Identity", icon="🪪", url_path="identity")
        )
        pages.append(
            st.Page(system_prompt.render, title="System prompt", icon="📝", url_path="system-prompt")
        )
        pages.append(
            st.Page(schedules.render, title="Schedules", icon="📅", url_path="schedules")
        )
        pages.append(
            st.Page(tools.render, title="Tools", icon="🛠️", url_path="tools")
        )
        pages.append(
            st.Page(telegram.render, title="Telegram", icon="💬", url_path="telegram")
        )
        pages.append(
            st.Page(browse.render, title="Browse", icon="📚", url_path="browse")
        )
        pages.append(
            st.Page(service.render, title="Service", icon="⚡", url_path="service")
        )
    pages.append(
        st.Page(install_settings.render, title="Install settings", icon="⚙️", url_path="install-settings")
    )

    pg = st.navigation(pages)
    pg.run()


_main()
