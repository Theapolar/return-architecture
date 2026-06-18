"""Home page — overview and quick links.

Will be the dashboard once per-agent pages exist. For now it's an
agent listing and a pointer toward setup if nothing is configured yet.
"""

from __future__ import annotations

import streamlit as st

from return_architecture import paths
from return_architecture.gui import helpers
from return_architecture.gui.pages import wizard


def render() -> None:
    st.title("Return Architecture")
    st.caption("Local agent runtime — control panel")

    st.write(f"**Install root**: `{paths.install_root()}`")

    secrets = helpers.load_install_secrets_raw()
    anthropic_set = helpers.provider_key_set(secrets, "anthropic")
    openai_set = helpers.provider_key_set(secrets, "openai")

    if not (anthropic_set or openai_set):
        st.warning(
            "No API keys are set yet. Go to **Install settings** in the "
            "sidebar to add at least one provider key before continuing."
        )
        return

    agents = helpers.list_agents()
    st.subheader("Agents")
    if not agents:
        st.info(
            "No agents yet. Per-agent setup is coming in the next "
            "GUI step; for now you can use the CLI: "
            "`return-architecture init <slug>`."
        )
        return
    for slug in agents:
        st.write(f"- `{slug}`")

    st.divider()
    if st.button("➕ Add another agent", key="_home_add_agent", type="primary"):
        wizard.start_add_agent()
