from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from interface_ai_challenge.domain.geometry import Size

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-5.6-sol"

ReasoningSummary = Literal["auto", "concise", "detailed"]


@dataclass(frozen=True)
class ModelSettings:
    """Which provider adapter and model discovery uses."""

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    reasoning_effort: str | None = "low"
    reasoning_summary: ReasoningSummary | None = None


@dataclass(frozen=True)
class RuntimeSettings:
    viewport: Size = field(default_factory=lambda: Size(width=1280, height=800))
    headless: bool = True
    enable_operator: bool = False
    operator_port: int | None = None
    intervention_timeout_seconds: float = 600.0
    # Live terminal output only. Evidence files are unaffected.
    console_events: bool = False
    console_model_decisions: bool = False


class LlmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    reasoning_effort: str | None = "low"
    reasoning_summary: ReasoningSummary | None = None


class BrowserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headless: bool = True
    viewport: Size = Size(width=1280, height=800)


class OperatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intervention_timeout_seconds: float = Field(default=600.0, gt=0)


class ConsoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: bool = False
    model_decisions: bool = False


class SystemConfig(BaseModel):
    """System settings file (`configs/system.json`). Secrets such as API keys stay in the environment."""

    model_config = ConfigDict(extra="forbid")

    llm: LlmConfig = LlmConfig()
    browser: BrowserConfig = BrowserConfig()
    operator: OperatorConfig = OperatorConfig()
    console: ConsoleConfig = ConsoleConfig()

    def llm_settings(self, provider: str | None = None, model: str | None = None) -> ModelSettings:
        return ModelSettings(
            provider=provider or self.llm.provider,
            model=model or self.llm.model,
            reasoning_effort=self.llm.reasoning_effort,
            reasoning_summary=self.llm.reasoning_summary,
        )

    def runtime(self, operator_port: int | None = None) -> RuntimeSettings:
        return RuntimeSettings(
            viewport=self.browser.viewport,
            headless=self.browser.headless,
            enable_operator=operator_port is not None,
            operator_port=operator_port,
            intervention_timeout_seconds=self.operator.intervention_timeout_seconds,
            console_events=self.console.events,
            console_model_decisions=self.console.model_decisions,
        )
