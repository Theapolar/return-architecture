"""Service control for the background daemon.

macOS:  writes a per-agent plist to ~/Library/LaunchAgents/ and manages it
        via launchctl (bootstrap / bootout / kickstart).
Linux:  writes a systemd user unit to ~/.config/systemd/user/ and manages it
        via ``systemctl --user`` (enable / restart / disable).

Windows is not supported.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from return_architecture import paths


# -- macOS / launchd helpers -------------------------------------------------

def _label(slug: str) -> str:
    return f"com.returnarchitecture.{slug}.daemon"


def _plist_path(slug: str) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{_label(slug)}.plist"


def _executable_path() -> Path:
    """Return path to the return-architecture binary in the active venv."""
    return Path(sys.executable).parent / "return-architecture"


def stdout_log(slug: str) -> Path:
    return paths.agent_logs_dir(slug) / "daemon-stdout.log"


def stderr_log(slug: str) -> Path:
    return paths.agent_logs_dir(slug) / "daemon-stderr.log"


# -- Linux / systemd helpers -------------------------------------------------

def _linux_unit_name(slug: str) -> str:
    return f"return-architecture-{slug}.service"


def _linux_unit_path(slug: str) -> Path:
    return Path.home() / ".config" / "systemd" / "user" / _linux_unit_name(slug)


def _linux_unit_content(slug: str) -> str:
    exe = _executable_path()
    install_root = paths.install_root()
    return (
        "[Unit]\n"
        f"Description=Return Architecture daemon ({slug})\n"
        "After=network-online.target\n"
        "\n"
        "[Service]\n"
        f"ExecStart={exe} daemon {slug}\n"
        f"Environment=RA_INSTALL_ROOT={install_root}\n"
        "Restart=on-failure\n"
        "RestartSec=10\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
    )


# -- Shared dataclass --------------------------------------------------------

@dataclass
class ServiceStatus:
    label: str
    config_path: Path   # plist file (macOS) or unit file (Linux)
    config_exists: bool
    loaded: bool
    pid: int | None
    raw_output: str


# -- Public API --------------------------------------------------------------

def install(slug: str) -> Path:
    """Write the service definition and start the daemon.

    Returns the path to the created config file (plist or unit file).
    """
    exe = _executable_path()
    if not exe.exists():
        raise FileNotFoundError(
            f"Can't find return-architecture executable at {exe}. "
            f"Run `uv sync` in the source repo, then try again."
        )

    system = platform.system()

    if system == "Darwin":
        paths.agent_logs_dir(slug).mkdir(parents=True, exist_ok=True)
        plist_path = _plist_path(slug)
        plist_path.parent.mkdir(parents=True, exist_ok=True)

        plist_content = _build_plist(
            label=_label(slug),
            executable=str(exe),
            slug=slug,
            install_root=str(paths.install_root()),
            stdout_path=str(stdout_log(slug)),
            stderr_path=str(stderr_log(slug)),
        )
        plist_path.write_text(plist_content, encoding="utf-8")

        # If a previous version is already loaded, unload it first and wait for
        # launchd to actually release the service -- bootstrap can fail with EIO
        # if the previous instance is still being torn down.
        if is_loaded(slug):
            _try_bootout(slug)
            import time
            for _ in range(25):
                if not is_loaded(slug):
                    break
                time.sleep(0.2)

        result = subprocess.run(
            ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"launchctl bootstrap failed (exit {result.returncode})"
                + (f": {msg}" if msg else "")
            )
        return plist_path

    elif system == "Linux":
        unit_path = _linux_unit_path(slug)
        unit_path.parent.mkdir(parents=True, exist_ok=True)
        unit_path.write_text(_linux_unit_content(slug), encoding="utf-8")

        _systemctl("daemon-reload")
        # enable creates the WantedBy symlink; restart starts (or restarts) the unit.
        _systemctl("enable", _linux_unit_name(slug))
        result = _systemctl("restart", _linux_unit_name(slug))
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"systemctl restart failed (exit {result.returncode})"
                + (f": {msg}" if msg else "")
            )
        return unit_path

    else:
        raise RuntimeError(
            f"service commands are not supported on {system}. "
            "macOS (launchd) and Linux (systemd) are supported."
        )


def is_loaded(slug: str) -> bool:
    """Return True if the daemon is currently running."""
    system = platform.system()
    if system == "Darwin":
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{_label(slug)}"],
            capture_output=True,
        )
        return proc.returncode == 0
    elif system == "Linux":
        proc = _systemctl("is-active", "--quiet", _linux_unit_name(slug))
        return proc.returncode == 0
    return False


def restart(slug: str) -> None:
    """Restart the running daemon in place.

    On macOS uses ``launchctl kickstart -k``, which terminates the current
    daemon process and lets launchd respawn it -- the new daemon re-reads
    config files on startup.

    On Linux uses ``systemctl --user restart``.

    Falls back to install() if the service is not currently loaded.
    """
    system = platform.system()

    if system == "Darwin":
        if not is_loaded(slug):
            install(slug)
            return
        result = subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{_label(slug)}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"launchctl kickstart failed (exit {result.returncode})"
                + (f": {msg}" if msg else "")
            )

    elif system == "Linux":
        if not is_loaded(slug):
            install(slug)
            return
        result = _systemctl("restart", _linux_unit_name(slug))
        if result.returncode != 0:
            msg = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"systemctl restart failed (exit {result.returncode})"
                + (f": {msg}" if msg else "")
            )

    else:
        raise RuntimeError(
            f"service commands are not supported on {system}. "
            "macOS (launchd) and Linux (systemd) are supported."
        )


def uninstall(slug: str) -> None:
    """Stop and remove the service definition."""
    system = platform.system()

    if system == "Darwin":
        _try_bootout(slug)
        plist_path = _plist_path(slug)
        if plist_path.exists():
            plist_path.unlink()

    elif system == "Linux":
        unit_name = _linux_unit_name(slug)
        _systemctl("disable", "--now", unit_name)
        unit_path = _linux_unit_path(slug)
        if unit_path.exists():
            unit_path.unlink()
        _systemctl("daemon-reload")

    else:
        raise RuntimeError(
            f"service commands are not supported on {system}. "
            "macOS (launchd) and Linux (systemd) are supported."
        )


def status(slug: str) -> ServiceStatus:
    """Return a ServiceStatus snapshot for the given agent."""
    system = platform.system()

    if system == "Darwin":
        plist_path = _plist_path(slug)
        label = _label(slug)
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
            capture_output=True,
            text=True,
        )
        loaded = proc.returncode == 0
        pid: int | None = None
        if loaded:
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.startswith("pid ="):
                    try:
                        pid = int(line.split("=")[1].strip())
                    except ValueError:
                        pid = None
                    break
        return ServiceStatus(
            label=label,
            config_path=plist_path,
            config_exists=plist_path.exists(),
            loaded=loaded,
            pid=pid,
            raw_output=(proc.stdout if loaded else proc.stderr),
        )

    elif system == "Linux":
        unit_name = _linux_unit_name(slug)
        unit_path = _linux_unit_path(slug)
        proc = _systemctl(
            "show", unit_name,
            "--property=ActiveState,MainPID,SubState",
        )
        props: dict[str, str] = {}
        for line in proc.stdout.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                props[k.strip()] = v.strip()
        loaded = props.get("ActiveState") == "active"
        main_pid = props.get("MainPID", "0")
        try:
            pid = int(main_pid) if main_pid not in ("0", "") else None
        except ValueError:
            pid = None
        return ServiceStatus(
            label=unit_name,
            config_path=unit_path,
            config_exists=unit_path.exists(),
            loaded=loaded,
            pid=pid,
            raw_output=proc.stdout,
        )

    else:
        raise RuntimeError(
            f"service commands are not supported on {system}. "
            "macOS (launchd) and Linux (systemd) are supported."
        )


def tail_logs(slug: str, lines: int = 40) -> tuple[str, str]:
    """Return (stdout_tail, stderr_tail).

    On macOS reads the plist-configured log files.
    On Linux returns journal output as *stdout_tail* and an empty string for
    *stderr_tail* -- the journal captures both streams together.
    """
    if platform.system() == "Linux":
        result = subprocess.run(
            [
                "journalctl", "--user",
                "-u", _linux_unit_name(slug),
                "-n", str(lines),
                "--no-pager",
                "--output=short",
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout, ""

    out = _tail_file(stdout_log(slug), lines)
    err = _tail_file(stderr_log(slug), lines)
    return out, err


# -- macOS internals ---------------------------------------------------------

def _tail_file(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    try:
        result = subprocess.run(
            ["tail", "-n", str(lines), str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return ""


def _try_bootout(slug: str) -> None:
    """Best-effort unload; ignore errors if the service is not loaded."""
    subprocess.run(
        ["launchctl", "bootout", f"gui/{os.getuid()}/{_label(slug)}"],
        capture_output=True,
    )


def _build_plist(
    *,
    label: str,
    executable: str,
    slug: str,
    install_root: str,
    stdout_path: str,
    stderr_path: str,
) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"'
        ' "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        "<dict>\n"
        "    <key>Label</key>\n"
        f"    <string>{label}</string>\n"
        "    <key>ProgramArguments</key>\n"
        "    <array>\n"
        f"        <string>{executable}</string>\n"
        "        <string>daemon</string>\n"
        f"        <string>{slug}</string>\n"
        "    </array>\n"
        "    <key>EnvironmentVariables</key>\n"
        "    <dict>\n"
        "        <key>RA_INSTALL_ROOT</key>\n"
        f"        <string>{install_root}</string>\n"
        "    </dict>\n"
        "    <key>RunAtLoad</key>\n"
        "    <true/>\n"
        "    <key>KeepAlive</key>\n"
        "    <true/>\n"
        "    <key>ThrottleInterval</key>\n"
        "    <integer>60</integer>\n"
        "    <key>StandardOutPath</key>\n"
        f"    <string>{stdout_path}</string>\n"
        "    <key>StandardErrorPath</key>\n"
        f"    <string>{stderr_path}</string>\n"
        "</dict>\n"
        "</plist>\n"
    )
