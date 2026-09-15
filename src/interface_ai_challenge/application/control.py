from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from interface_ai_challenge.domain.control import ActorKind, ControlSnapshot, ControlState, ControlToken
from interface_ai_challenge.domain.errors import ControlError

AUTOMATION_ID = "automation"


class SessionControl:
    """Single source of truth for who may act on the live session.

    Every action and every transfer takes the same lock, so a transfer waits for an in-flight
    action to finish and bumps the epoch, which invalidates tokens held by the previous owner.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._state = ControlState.AUTOMATION
        self._owner_id: str | None = AUTOMATION_ID
        self._epoch = 0

    def snapshot(self) -> ControlSnapshot:
        return ControlSnapshot(state=self._state, owner_id=self._owner_id, epoch=self._epoch)

    def automation_token(self) -> ControlToken:
        if self._state is not ControlState.AUTOMATION:
            raise ControlError(f"automation does not own the session (state={self._state})")
        return ControlToken(actor=ActorKind.AUTOMATION, actor_id=AUTOMATION_ID, epoch=self._epoch)

    @asynccontextmanager
    async def acting(self, token: ControlToken) -> AsyncIterator[None]:
        async with self._lock:
            self._ensure_owner(token)
            yield

    async def pause_for_human(self) -> ControlSnapshot:
        async with self._lock:
            self._require(ControlState.AUTOMATION)
            self._transition(ControlState.WAITING_FOR_HUMAN, owner_id=None)
            return self.snapshot()

    async def claim(self, operator_id: str) -> ControlToken:
        async with self._lock:
            if not (self._state is ControlState.HUMAN and self._owner_id == operator_id):
                self._require(ControlState.WAITING_FOR_HUMAN)
                self._transition(ControlState.HUMAN, owner_id=operator_id)
            return ControlToken(actor=ActorKind.HUMAN, actor_id=operator_id, epoch=self._epoch)

    async def release_to_automation(self, operator_id: str) -> ControlSnapshot:
        async with self._lock:
            self._require(ControlState.HUMAN)
            if self._owner_id != operator_id:
                raise ControlError("only the operator in control can hand control back")
            self._transition(ControlState.AUTOMATION, owner_id=AUTOMATION_ID)
            return self.snapshot()

    async def terminate(self) -> ControlSnapshot:
        async with self._lock:
            self._transition(ControlState.TERMINATED, owner_id=None)
            return self.snapshot()

    def _ensure_owner(self, token: ControlToken) -> None:
        expected = ControlState.AUTOMATION if token.actor is ActorKind.AUTOMATION else ControlState.HUMAN
        if self._state is not expected or token.actor_id != self._owner_id or token.epoch != self._epoch:
            raise ControlError("stale or unauthorized control token")

    def _require(self, state: ControlState) -> None:
        if self._state is not state:
            raise ControlError(f"expected state {state}, session is {self._state}")

    def _transition(self, state: ControlState, owner_id: str | None) -> None:
        self._state = state
        self._owner_id = owner_id
        self._epoch += 1
