"""Config loading and validation for install and per-agent settings."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from return_architecture import paths


# ── Install-wide ────────────────────────────────────────────────────────────

class InstallSection(BaseModel):
    created_at: str
    default_agent: str | None = None


class GuiSection(BaseModel):
    port: int = 7878
    address: str = "127.0.0.1"
    open_browser_on_start: bool = True


class UiSection(BaseModel):
    show_cost_estimates: bool = True


class LogsSection(BaseModel):
    retention_days: int = 90


class InstallConfig(BaseModel):
    install: InstallSection
    gui: GuiSection = Field(default_factory=GuiSection)
    ui: UiSection = Field(default_factory=UiSection)
    logs: LogsSection = Field(default_factory=LogsSection)


class ProviderSecrets(BaseModel):
    anthropic: str | None = None
    openai: str | None = None
    gemini: str | None = None


class InstallSecrets(BaseModel):
    providers: ProviderSecrets


# ── Per-agent ───────────────────────────────────────────────────────────────

Provider = Literal["anthropic", "openai", "gemini"]


class AgentSection(BaseModel):
    name: str
    slug: str
    # The person this agent is in relationship with. Optional; used where a
    # channel needs to address or refer to the human by name (e.g. the Presence
    # app). Falls back to neutral wording when unset.
    human_name: str | None = None


class ModelSection(BaseModel):
    provider: Provider
    name: str
    max_tokens: int = 4096
    # None means "let the provider use its default" — useful for models
    # (e.g. some OpenAI reasoning models) that only accept the default.
    temperature: float | None = None
    # Sampling knobs. top_p applies to all providers; top_k applies to
    # anthropic + gemini (openai ignores it).
    top_p: float | None = None
    top_k: int | None = None
    # Gemini-only. 0 disables thinking (2.5 Flash), -1 lets the model
    # choose dynamically, a positive number is a hard token cap. None
    # means provider default (dynamic for 2.5 models).
    thinking_budget: int | None = None
    # Provider-native built-in tools to enable alongside function-call tools.
    # Gemini 3 names: google_search, url_context, code_execution, google_maps,
    # file_search, computer_use. Other providers ignore this. Off by default —
    # each agent opts in per tool.
    native_tools: list[str] = Field(default_factory=list)


class BehaviorSection(BaseModel):
    silence_allowed: bool = True
    max_self_scheduled_jobs_per_day: int = 5
    # When > 0, each new session is pre-filled with this many of the agent's
    # most recent memory entries as real chat history. The agent "arrives"
    # with prior turns already in context, instead of only retrieving them
    # via semantic recall. 0 = current behavior (empty session).
    seed_chat_history_from_memory: int = 0
    # How many tool-call rounds a single turn may chain before the runtime
    # stops with "(stopped: tool loop limit reached)". 8 is fine for chat;
    # raise it for agents that do multi-step work like editing several files.
    max_tool_loops: int = 8
    # Upper bound on how many recent messages are sent to the provider each
    # turn. The live conversation otherwise grows for the whole lifetime of the
    # daemon and the *entire* transcript is re-sent on every API call (and every
    # tool loop), so cost climbs without limit. Windowing caps that: older turns
    # drop out of the live context but are NOT lost — continuity still comes from
    # memory recall + seed_chat_history_from_memory. 0 = unlimited (old behavior).
    max_context_messages: int = 40


class ArtifactExchangeSection(BaseModel):
    enabled: bool = True
    mediator_provider: Provider = "anthropic"
    mediator_model: str = "claude-sonnet-4-6"
    agent_max_tokens: int = 600
    mediator_max_tokens: int = 600


class ReflectiveReviewSection(BaseModel):
    enabled: bool = False
    # The analyzer must run on a *different* model than the agent. Defaults to
    # Anthropic so an OpenAI-backed agent gets genuine separation out of the box.
    analyzer_provider: Provider = "anthropic"
    analyzer_model: str = "claude-sonnet-4-6"
    threshold_days: int = 14
    threshold_messages: int = 300
    # Cap on how far back context is gathered, so a long gap can't balloon the prompt.
    max_lookback_days: int = 45
    max_tokens: int = 1200


class ToolsSection(BaseModel):
    enabled: list[str] = Field(default_factory=lambda: [
        "no_response", "send_to_human_telegram",
        "artifact_delete_reaction", "artifact_share_more",
        "tag_item", "write_letter",
    ])
    artifact_exchange: ArtifactExchangeSection = Field(default_factory=ArtifactExchangeSection)


class ScheduleEntry(BaseModel):
    enabled: bool = False
    cron: str
    prompt: str
    kind: str = "regular"


class MCPServerConfig(BaseModel):
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class MCPSection(BaseModel):
    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    agent: AgentSection
    model: ModelSection
    behavior: BehaviorSection = Field(default_factory=BehaviorSection)
    tools: ToolsSection = Field(default_factory=ToolsSection)
    schedules: dict[str, ScheduleEntry] = Field(default_factory=dict)
    mcp: MCPSection = Field(default_factory=MCPSection)
    reflective_review: ReflectiveReviewSection = Field(default_factory=ReflectiveReviewSection)


# ── Loaders ──────────────────────────────────────────────────────────────────

def _read_toml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected config at {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_install_config() -> InstallConfig:
    return InstallConfig.model_validate(_read_toml(paths.install_config_path()))


def load_install_secrets() -> InstallSecrets:
    return InstallSecrets.model_validate(_read_toml(paths.install_secrets_path()))


def load_agent_config(slug: str) -> AgentConfig:
    return AgentConfig.model_validate(_read_toml(paths.agent_config_path(slug)))


def update_agent_config_value(slug: str, section: str, key: str, value) -> None:
    """Persist a single key in one section of an agent's config TOML.

    Reads the raw file, sets [section].key = value, writes it back —
    leaving every other field untouched. Used by tools that let an agent
    adjust its own settings (e.g. set_temperature) so the change survives
    a service restart.
    """
    import tomli_w

    path = paths.agent_config_path(slug)
    data = _read_toml(path)
    data.setdefault(section, {})[key] = value
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def load_system_prompt(slug: str) -> str:
    path = paths.agent_system_prompt_path(slug)
    return path.read_text(encoding="utf-8").strip()
