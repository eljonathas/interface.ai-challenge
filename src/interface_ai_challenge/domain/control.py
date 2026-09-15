from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ControlState(StrEnum):
    AUTOMATION = "automation"
    WAITING_FOR_HUMAN = "waiting_for_human"
    HUMAN = "human"
    TERMINATED = "terminated"


class ActorKind(StrEnum):
    AUTOMATION = "automation"
    HUMAN = "human"


class ControlToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    actor: ActorKind
    actor_id: str
    epoch: int


class ControlSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    state: ControlState
    owner_id: str | None
    epoch: int
