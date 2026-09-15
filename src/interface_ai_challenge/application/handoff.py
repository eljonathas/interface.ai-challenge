from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from interface_ai_challenge.application.control import SessionControl
from interface_ai_challenge.domain.errors import SurfaceError
from interface_ai_challenge.domain.intervention import InterventionDecision, InterventionRequest
from interface_ai_challenge.ports.evidence import EvidenceSink
from interface_ai_challenge.ports.intervention import InterventionChannel
from interface_ai_challenge.ports.surface import ScreenCapture


@dataclass(frozen=True)
class EscalationContext:
    mode: Literal["discovery", "replay"]
    subject: str
    step_id: str | None
    reason: str
    resume_behavior: str
    evidence_reason: str | None = None
    expected: dict[str, Any] = field(default_factory=dict)
    observed: dict[str, Any] = field(default_factory=dict)


class InterventionCoordinator:
    def __init__(
        self,
        run_id: str,
        control: SessionControl,
        channel: InterventionChannel | None,
        evidence_screen: ScreenCapture,
        evidence: EvidenceSink,
        timeout_seconds: float,
    ) -> None:
        self._run_id = run_id
        self._control = control
        self._channel = channel
        self._screen = evidence_screen
        self._evidence = evidence
        self._timeout_seconds = timeout_seconds
        self.interventions = 0

    @property
    def available(self) -> bool:
        return self._channel is not None

    async def escalate(self, context: EscalationContext) -> InterventionDecision:
        if self._channel is None:
            raise RuntimeError("no intervention channel configured")
        await self._control.pause_for_human()
        request = InterventionRequest(
            id=f"int_{uuid.uuid4().hex[:12]}",
            run_id=self._run_id,
            mode=context.mode,
            subject=context.subject,
            step_id=context.step_id,
            reason=context.reason,
            expected=context.expected,
            observed=context.observed,
            screenshot_ref=await self._capture_evidence(),
            resume_behavior=context.resume_behavior,
            created_at=datetime.now(UTC),
        )
        recorded_request = request.model_dump(mode="json")
        if context.evidence_reason is not None:
            recorded_request["reason"] = context.evidence_reason
        self._evidence.emit("intervention_requested", recorded_request)
        await self._channel.publish(request)
        decision = await self._channel.wait_for_decision(request.id, self._timeout_seconds)
        self.interventions += 1
        if decision is not InterventionDecision.RESUME:
            await self._control.terminate()
        self._evidence.emit(
            "intervention_resolved",
            {"intervention_id": request.id, "decision": decision, "control": self._control.snapshot().model_dump()},
        )
        return decision

    async def _capture_evidence(self) -> str | None:
        try:
            screenshot = await self._screen.capture()
        except SurfaceError:
            self._evidence.emit("screenshot_omitted", {"reason": "capture or redaction unavailable"})
            return None
        return self._evidence.attach_image("intervention", screenshot.png)
