from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class InterventionDecision(StrEnum):
    RESUME = "resume"
    ABORT = "abort"
    TIMEOUT = "timeout"


class InterventionRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    run_id: str
    mode: Literal["discovery", "replay"]
    subject: str
    step_id: str | None
    reason: str
    expected: dict[str, Any] = Field(default_factory=dict)
    observed: dict[str, Any] = Field(default_factory=dict)
    screenshot_ref: str | None = None
    resume_behavior: str
    created_at: datetime
