from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FailureCode(StrEnum):
    INPUT_INVALID = "input_invalid"
    ARTIFACT_INCOMPATIBLE = "artifact_incompatible"
    POLICY_VIOLATION = "policy_violation"
    PERMISSION_DENIED = "permission_denied"
    TARGET_NOT_FOUND = "target_not_found"
    TARGET_AMBIGUOUS = "target_ambiguous"
    ACTION_FAILED = "action_failed"
    POSTCONDITION_TIMEOUT = "postcondition_timeout"
    AMBIGUOUS_STATE = "ambiguous_state"
    CHECKPOINT_FAILED = "checkpoint_failed"
    EXTRACTION_FAILED = "extraction_failed"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    ESCALATION_UNAVAILABLE = "escalation_unavailable"
    INTERVENTION_TIMEOUT = "intervention_timeout"
    INTERVENTION_ABORTED = "intervention_aborted"
    SESSION_LOST = "session_lost"


class RunSuccess(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["success"] = "success"
    run_id: str
    capability_id: str
    outputs: dict[str, str]
    assisted: bool = False
    evidence_ref: str | None = None


class RunBusinessOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["business_outcome"] = "business_outcome"
    run_id: str
    capability_id: str
    code: str
    step_id: str | None = None
    assisted: bool = False
    evidence_ref: str | None = None


class RunFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["failure"] = "failure"
    run_id: str
    capability_id: str
    code: FailureCode
    message: str
    step_id: str | None = None
    expected: dict[str, Any] = Field(default_factory=dict)
    observed: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False
    assisted: bool = False
    evidence_ref: str | None = None


RunResult = Annotated[RunSuccess | RunBusinessOutcome | RunFailure, Field(discriminator="status")]
