from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from interface_ai_challenge.domain.predicates import Predicate
from interface_ai_challenge.domain.results import FailureCode
from interface_ai_challenge.domain.steps import Origin


class HandlerCategory(StrEnum):
    BUSINESS_OUTCOME = "business_outcome"
    RECOVERABLE = "recoverable"
    INTERVENTION = "intervention"
    TERMINAL = "terminal"


class BusinessOutcomeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["business_outcome"] = "business_outcome"
    code: str

    @property
    def category(self) -> HandlerCategory:
        return HandlerCategory.BUSINESS_OUTCOME


class RecoverByClickResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["recover_click"] = "recover_click"
    target_ref: str

    @property
    def category(self) -> HandlerCategory:
        return HandlerCategory.RECOVERABLE


class RecoverByWaitingResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["recover_wait"] = "recover_wait"
    extra_wait_ms: int = Field(gt=0, le=60_000)

    @property
    def category(self) -> HandlerCategory:
        return HandlerCategory.RECOVERABLE


class EscalateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["escalate"] = "escalate"
    reason: str

    @property
    def category(self) -> HandlerCategory:
        return HandlerCategory.INTERVENTION


class FailResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["fail"] = "fail"
    code: FailureCode

    @property
    def category(self) -> HandlerCategory:
        return HandlerCategory.TERMINAL


HandlerResponse = Annotated[
    BusinessOutcomeResponse | RecoverByClickResponse | RecoverByWaitingResponse | EscalateResponse | FailResponse,
    Field(discriminator="kind"),
]


class Handler(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    description: str
    when: Predicate
    response: HandlerResponse
    priority: int
    max_uses: int = Field(default=1, ge=1, le=5)
    origin: Origin = Origin.AUTHORED

    @property
    def category(self) -> HandlerCategory:
        return self.response.category
