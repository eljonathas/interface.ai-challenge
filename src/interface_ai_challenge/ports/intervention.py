from __future__ import annotations

from typing import Protocol

from interface_ai_challenge.domain.intervention import InterventionDecision, InterventionRequest


class InterventionChannel(Protocol):
    async def publish(self, request: InterventionRequest) -> None: ...

    async def wait_for_decision(self, request_id: str, timeout_seconds: float) -> InterventionDecision: ...
