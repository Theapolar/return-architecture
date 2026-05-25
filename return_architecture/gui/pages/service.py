"""Service page -- control the background daemon."""

from __future__ import annotations

import platform

import streamlit as st

from return_architecture import service as ra_service

_IS_LINUX = platform.system() == "Linux"
_CONFIG_LABEL = "Unit file" if _IS_LINUX else "Plist"
_INSTALL_HINT = (
    "Write the systemd unit file and start the daemon."
    if _IS_LINUX
    else "Write the plist and start the daemon."
)
_UNINSTALL_REMOVES = (
    "the unit file will be removed"
    if _IS_LINUX
    else "the plist will be removed"
)


def render() -> None:
    slug = st.session_state.get("agent_slug")
    if not slug:
        st.warning("No agent selected. Use the sidebar.")
        return

    st.title(f"Service -- {slug}")
    st.caption(
        "The background daemon that runs Telegram + scheduler for this agent. "
        "When installed, it auto-starts on boot and respawns on crash."
    )

    try:
        status = ra_service.status(slug)
    except RuntimeError as e:
        st.info(str(e))
        return

    _render_status(status)
    st.divider()
    _render_actions(slug, status)
    st.divider()
    _render_logs(slug)


# -- Status ------------------------------------------------------------------

def _render_status(status: ra_service.ServiceStatus) -> None:
    st.subheader("Status")

    cols = st.columns(2)
    with cols[0]:
        if status.loaded:
            pid_part = f" (PID {status.pid})" if status.pid else ""
            st.success(f"loaded {pid_part}")
        else:
            st.warning("not loaded")
        st.markdown(f"**Label**: `{status.label}`")
    with cols[1]:
        st.markdown(f"**{_CONFIG_LABEL}**: `{status.config_path}`")
        st.markdown(
            f"**{_CONFIG_LABEL}**: {'exists' if status.config_exists else 'missing'}"
        )


# -- Actions -----------------------------------------------------------------

def _render_actions(slug: str, status: ra_service.ServiceStatus) -> None:
    st.subheader("Actions")

    cols = st.columns([1, 1, 1, 4])

    if cols[0].button(
        "Install",
        disabled=status.loaded,
        help=(
            "Already loaded -- use Restart to apply config changes."
            if status.loaded
            else _INSTALL_HINT
        ),
        key="_svc_install",
    ):
        try:
            with st.spinner("Installing..."):
                ra_service.install(slug)
            st.success("Service installed and running.")
            st.rerun()
        except (RuntimeError, FileNotFoundError) as e:
            st.error(f"Install failed: {e}")

    if cols[1].button(
        "Restart",
        disabled=not status.loaded,
        help=(
            "Terminates the running daemon process; it respawns and "
            "re-reads your config."
            if status.loaded
            else "Service is not loaded -- use Install."
        ),
        key="_svc_restart",
    ):
        try:
            with st.spinner("Restarting..."):
                ra_service.restart(slug)
            st.success("Service restarted.")
            st.rerun()
        except (RuntimeError, FileNotFoundError) as e:
            st.error(f"Restart failed: {e}")

    can_uninstall = status.loaded or status.config_exists
    if cols[2].button(
        "Uninstall",
        disabled=not can_uninstall,
        help=(
            "Stops the daemon and removes the service definition. Your config, "
            "memory, letters and items are NOT affected; reinstall any time."
            if can_uninstall
            else "Nothing to uninstall."
        ),
        key="_svc_uninstall_btn",
    ):
        st.session_state["_confirm_uninstall"] = True
        st.rerun()

    if st.session_state.get("_confirm_uninstall"):
        st.warning(
            f"Uninstall the **{slug}** service? The daemon will stop running and "
            f"{_UNINSTALL_REMOVES}. Your config, memory, items, letters, and "
            "artifacts are NOT affected -- you can reinstall any time."
        )
        confirm_cols = st.columns([1, 1, 5])
        if confirm_cols[0].button("Yes, uninstall", key="_confirm_uninstall_yes"):
            try:
                with st.spinner("Uninstalling..."):
                    ra_service.uninstall(slug)
                st.success("Service uninstalled.")
            except RuntimeError as e:
                st.error(f"Uninstall failed: {e}")
            st.session_state["_confirm_uninstall"] = False
            st.rerun()
        if confirm_cols[1].button("Cancel", key="_confirm_uninstall_no"):
            st.session_state["_confirm_uninstall"] = False
            st.rerun()


# -- Logs --------------------------------------------------------------------

def _render_logs(slug: str) -> None:
    st.subheader("Recent output")
    if _IS_LINUX:
        st.caption(
            "Last N lines from the systemd journal for this agent's daemon. "
            "Stdout and stderr are captured together."
        )
    else:
        st.caption(
            "Last N lines from the daemon's stdout and stderr log files. "
            "Stdout includes startup messages and scheduler pings; stderr "
            "shows errors and tracebacks."
        )

    cols = st.columns([1, 1, 5])
    lines = cols[0].number_input(
        "Lines per log",
        min_value=10, max_value=500,
        value=40, step=10,
        key="_logs_lines",
        label_visibility="collapsed",
    )
    cols[0].caption("lines")
    if cols[1].button("Refresh", key="_logs_refresh"):
        st.rerun()

    try:
        stdout, stderr = ra_service.tail_logs(slug, lines=int(lines))
    except RuntimeError as e:
        st.info(str(e))
        return

    if _IS_LINUX:
        st.markdown("**Journal log**")
        if stdout:
            st.code(stdout, language="text")
        else:
            st.caption("(empty)")
    else:
        st.markdown("**stdout**")
        if stdout:
            st.code(stdout, language="text")
        else:
            st.caption("(empty)")

        st.markdown("**stderr**")
        if stderr:
            st.code(stderr, language="text")
        else:
            st.caption("(empty)")
