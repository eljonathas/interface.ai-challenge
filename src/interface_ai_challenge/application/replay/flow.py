from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from interface_ai_challenge.domain.results import FailureCode, RunBusinessOutcome, RunFailure, RunSuccess


@dataclass(frozen=True)
class Continue:
    pass


@dataclass(frozen=True)
class RestartFrom:
    index: int
    reason: str


@dataclass(frozen=True)
class Stop:
    result: RunBusinessOutcome | RunFailure


StepOutcome = Continue | RestartFrom | Stop


@dataclass(frozen=True)
class KeepWaiting:
    extra_ms: int = 0


@dataclass(frozen=True)
class HumanResumed:
    pass


HandlerEffect = KeepWaiting | HumanResumed | Stop


class ResultFactory:
    def __init__(self, run_id: str, capability_id: str, assisted: Any) -> None:
        self._run_id = run_id
        self._capability_id = capability_id
        self._assisted = assisted

    def success(self, outputs: dict[str, str]) -> RunSuccess:
        return RunSuccess(
            run_id=self._run_id, capability_id=self._capability_id, outputs=outputs, assisted=self._assisted()
        )

    def outcome(self, code: str, step_id: str | None) -> RunBusinessOutcome:
        return RunBusinessOutcome(
            run_id=self._run_id,
            capability_id=self._capability_id,
            code=code,
            step_id=step_id,
            assisted=self._assisted(),
        )

    def failure(
        self,
        code: FailureCode,
        message: str,
        step_id: str | None = None,
        expected: dict[str, Any] | None = None,
        observed: dict[str, Any] | None = None,
        retryable: bool = False,
    ) -> RunFailure:
        return RunFailure(
            run_id=self._run_id,
            capability_id=self._capability_id,
            code=code,
            message=message,
            step_id=step_id,
            expected=expected or {},
            observed=observed or {},
            retryable=retryable,
            assisted=self._assisted(),
        )
