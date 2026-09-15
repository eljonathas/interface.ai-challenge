from __future__ import annotations

import asyncio
from collections.abc import Callable

from interface_ai_challenge.domain.intervention import InterventionDecision, InterventionRequest


class InMemoryInterventionInbox:
    """Routes intervention requests to operators attached to this process (console and CLI)."""

    def __init__(self, announce: Callable[[InterventionRequest], None] | None = None) -> None:
        self._announce = announce
        self._pending: dict[str, InterventionRequest] = {}
        self._decisions: dict[str, asyncio.Future[InterventionDecision]] = {}

    async def publish(self, request: InterventionRequest) -> None:
        self._pending[request.id] = request
        self._decisions[request.id] = asyncio.get_running_loop().create_future()
        if self._announce is not None:
            self._announce(request)

    async def wait_for_decision(self, request_id: str, timeout_seconds: float) -> InterventionDecision:
        decision = self._decisions[request_id]
        try:
            return await asyncio.wait_for(asyncio.shield(decision), timeout_seconds)
        except TimeoutError:
            return InterventionDecision.TIMEOUT
        finally:
            self._pending.pop(request_id, None)
            self._decisions.pop(request_id, None)

    def current(self) -> InterventionRequest | None:
        return next(iter(self._pending.values()), None)

    def decide(self, request_id: str, decision: InterventionDecision) -> bool:
        future = self._decisions.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(decision)
        return True
