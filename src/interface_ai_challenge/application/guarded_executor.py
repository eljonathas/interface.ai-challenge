from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from interface_ai_challenge.application.control import SessionControl
from interface_ai_challenge.application.policy_guard import ActionIntent, PolicyGuard
from interface_ai_challenge.domain.control import ControlToken
from interface_ai_challenge.domain.errors import PolicyViolationError
from interface_ai_challenge.ports.evidence import EvidenceSink

T = TypeVar("T")


class GuardedExecutor:
    """The only path to side effects: checks control ownership and policy under the control lock."""

    def __init__(self, control: SessionControl, guard: PolicyGuard, evidence: EvidenceSink) -> None:
        self._control = control
        self._guard = guard
        self._evidence = evidence

    async def run(self, token: ControlToken, intent: ActionIntent, operation: Callable[[], Awaitable[T]]) -> T:
        async with self._control.acting(token):
            decision = self._guard.evaluate(intent, token.actor)
            self._evidence.emit(
                "policy_decision",
                {
                    "actor": token.actor,
                    "actor_id": token.actor_id,
                    "control_epoch": token.epoch,
                    "intent": intent.model_dump(mode="json"),
                    "allowed": decision.allowed,
                    "reason": decision.reason,
                },
            )
            if not decision.allowed:
                raise PolicyViolationError(decision.reason)
            return await operation()
