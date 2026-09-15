from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from interface_ai_challenge.domain.predicates import Predicate
from interface_ai_challenge.domain.values import ValueRef


class Effect(StrEnum):
    READ = "read"
    NAVIGATION = "navigation"
    REVERSIBLE_INPUT = "reversible_input"
    SUBMIT = "submit"
    IRREVERSIBLE = "irreversible"
    UNKNOWN = "unknown"


class Origin(StrEnum):
    DISCOVERED = "discovered"
    AUTHORED = "authored"
    COMPILED = "compiled"
    HUMAN = "human"


class NavigateAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["navigate"] = "navigate"
    base_url: ValueRef
    path: str


class ClickAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["click"] = "click"
    target_ref: str


class FillAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["fill"] = "fill"
    target_ref: str
    value: ValueRef


class PressKeyAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["press_key"] = "press_key"
    target_ref: str
    key: Literal["Enter", "Tab", "Escape"]


StepAction = Annotated[
    NavigateAction | ClickAction | FillAction | PressKeyAction,
    Field(discriminator="kind"),
]


class Step(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    action: StepAction
    precondition: Predicate | None = None
    postcondition: Predicate
    effect: Effect
    timeout_ms: int = Field(default=10_000, gt=0)
    max_attempts: int = Field(default=1, ge=1, le=3)
    resume_boundary: bool = False
    origin: Origin

    def target_ref(self) -> str | None:
        return getattr(self.action, "target_ref", None)
