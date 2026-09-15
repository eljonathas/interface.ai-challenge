from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from interface_ai_challenge.domain.computer import ComputerCommand
from interface_ai_challenge.domain.geometry import Point, Screenshot


@dataclass(frozen=True)
class OutputSpec:
    name: str
    description: str


@dataclass(frozen=True)
class AgentTask:
    goal: str
    outputs: tuple[OutputSpec, ...]
    input_placeholders: tuple[str, ...]
    identity_placeholders: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportOutputSignal:
    call_id: str
    output: str
    point: Point


@dataclass(frozen=True)
class ReportIdentitySignal:
    """The model points at where the opened record displays an identifying input value."""

    call_id: str
    input: str
    point: Point


@dataclass(frozen=True)
class FinishSignal:
    call_id: str
    summary: str


@dataclass(frozen=True)
class RequestHelpSignal:
    call_id: str
    reason: str


AgentSignal = ReportOutputSignal | ReportIdentitySignal | FinishSignal | RequestHelpSignal


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class AgentTurn:
    commands: tuple[ComputerCommand, ...]
    signals: tuple[AgentSignal, ...]
    safety_checks: tuple[str, ...] = ()
    message: str | None = None
    # Provider reasoning summaries: shown live when enabled, never persisted.
    reasoning: tuple[str, ...] = ()
    usage: TokenUsage = TokenUsage()
    request_id: str | None = None
    # The provider needs a screenshot round trip before it can decide again.
    # Commands and signals in this turn must not be executed.
    observation_required: bool = False


@dataclass
class AgentFeedback:
    screenshot: Screenshot
    signal_replies: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    acknowledge_safety_checks: bool = False


class ComputerUseAgent(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    async def begin(self, task: AgentTask, screenshot: Screenshot) -> AgentTurn: ...

    async def respond(self, feedback: AgentFeedback) -> AgentTurn: ...


@dataclass(frozen=True)
class GoalParameter:
    name: str
    description: str
    value: str
    identifies_record: bool


@dataclass(frozen=True)
class GoalOutput:
    name: str
    description: str
    kind: Literal["text", "decimal_money"]


@dataclass(frozen=True)
class GoalInterpretation:
    """Model proposal of the contract implied by a free-text goal. Validated in code before use."""

    capability_id: str
    description: str
    parameters: tuple[GoalParameter, ...]
    outputs: tuple[GoalOutput, ...]
    clarification: str | None = None


class GoalInterpreter(Protocol):
    async def interpret(self, goal: str) -> GoalInterpretation: ...
