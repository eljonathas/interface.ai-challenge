from __future__ import annotations

from collections import Counter
from typing import Any

from interface_ai_challenge.application.control import SessionControl
from interface_ai_challenge.application.guarded_executor import GuardedExecutor
from interface_ai_challenge.application.handoff import EscalationContext, InterventionCoordinator
from interface_ai_challenge.application.policy_guard import ActionIntent, EffectClassifier
from interface_ai_challenge.application.replay.flow import HandlerEffect, HumanResumed, KeepWaiting, ResultFactory, Stop
from interface_ai_challenge.application.target_resolution import NamedTargetResolver
from interface_ai_challenge.domain.errors import ControlError, PolicyViolationError, SurfaceError
from interface_ai_challenge.domain.handlers import (
    BusinessOutcomeResponse,
    EscalateResponse,
    FailResponse,
    Handler,
    RecoverByClickResponse,
    RecoverByWaitingResponse,
)
from interface_ai_challenge.domain.intervention import InterventionDecision
from interface_ai_challenge.domain.results import FailureCode
from interface_ai_challenge.domain.steps import Step
from interface_ai_challenge.domain.values import ValueResolver
from interface_ai_challenge.ports.evidence import EvidenceSink
from interface_ai_challenge.ports.surface import ElementActions

_RESUME_BEHAVIOR = (
    "Automation takes a fresh observation; if the interrupted step's condition already holds it continues, "
    "otherwise it restarts from the last declared safe boundary. It never blindly advances."
)


class HandlerApplier:
    def __init__(
        self,
        capability_id: str,
        resolver: NamedTargetResolver,
        actions: ElementActions,
        executor: GuardedExecutor,
        control: SessionControl,
        classifier: EffectClassifier,
        coordinator: InterventionCoordinator,
        values: ValueResolver,
        results: ResultFactory,
        evidence: EvidenceSink,
    ) -> None:
        self._capability_id = capability_id
        self._resolver = resolver
        self._actions = actions
        self._executor = executor
        self._control = control
        self._classifier = classifier
        self._coordinator = coordinator
        self._values = values
        self._results = results
        self._evidence = evidence
        self._uses: Counter[str] = Counter()

    @property
    def can_escalate(self) -> bool:
        return self._coordinator.available

    async def apply(self, handler: Handler, step: Step | None, expected: dict[str, Any] | None = None) -> HandlerEffect:
        step_id = step.id if step else None
        self._uses[handler.id] += 1
        self._evidence.emit(
            "handler_triggered",
            {"handler_id": handler.id, "category": handler.category, "step_id": step_id, "use": self._uses[handler.id]},
        )
        if self._uses[handler.id] > handler.max_uses:
            return Stop(
                self._results.failure(
                    FailureCode.RECOVERY_EXHAUSTED,
                    f"handler '{handler.id}' was needed more than {handler.max_uses} time(s)",
                    step_id,
                    expected={"max_uses": handler.max_uses},
                    observed={"handler": handler.id},
                )
            )
        match handler.response:
            case BusinessOutcomeResponse(code=code):
                return Stop(self._results.outcome(code, step_id))
            case FailResponse(code=code):
                return Stop(
                    self._results.failure(
                        code, handler.description, step_id, expected=expected, observed={"handler": handler.id}
                    )
                )
            case RecoverByWaitingResponse(extra_wait_ms=extra):
                return KeepWaiting(extra_ms=extra)
            case RecoverByClickResponse(target_ref=ref):
                return await self._click(ref, handler, step_id)
            case EscalateResponse(reason=reason):
                return await self.escalate(reason, step, expected=expected, observed={"handler": handler.id})
        raise TypeError(f"unsupported handler response {handler.response!r}")

    async def escalate(
        self,
        reason: str,
        step: Step | None,
        expected: dict[str, Any] | None = None,
        observed: dict[str, Any] | None = None,
    ) -> HandlerEffect:
        step_id = step.id if step else None
        if not self._coordinator.available:
            return Stop(
                self._results.failure(
                    FailureCode.ESCALATION_UNAVAILABLE,
                    f"human intervention required ({reason}) but no operator channel is configured",
                    step_id,
                    expected=expected,
                    observed=observed,
                )
            )
        decision = await self._coordinator.escalate(
            EscalationContext(
                mode="replay",
                subject=self._capability_id,
                step_id=step_id,
                reason=reason,
                resume_behavior=_RESUME_BEHAVIOR,
                expected=expected or {},
                observed=observed or {},
            )
        )
        if decision is InterventionDecision.RESUME:
            return HumanResumed()
        code = (
            FailureCode.INTERVENTION_ABORTED
            if decision is InterventionDecision.ABORT
            else FailureCode.INTERVENTION_TIMEOUT
        )
        return Stop(self._results.failure(code, f"intervention ended with '{decision}'", step_id))

    async def _click(self, ref: str, handler: Handler, step_id: str | None) -> HandlerEffect:
        resolution = await self._resolver.resolve(ref, self._values)
        candidate = resolution.unique
        if candidate is None:
            count = len(resolution.candidates)
            return Stop(
                self._results.failure(
                    FailureCode.TARGET_AMBIGUOUS if count else FailureCode.TARGET_NOT_FOUND,
                    f"recovery control '{ref}' of handler '{handler.id}' did not resolve to exactly one element",
                    step_id,
                    expected={"unique_target": ref},
                    observed={"candidate_count": count},
                )
            )
        role, name = self._resolver.control_identity(ref)
        intent = ActionIntent(command="click", effect=self._classifier.for_control(role, name), control_name=name)
        try:
            await self._executor.run(
                self._control.automation_token(), intent, lambda: self._actions.click(candidate, 5000)
            )
        except PolicyViolationError as error:
            return Stop(self._results.failure(FailureCode.POLICY_VIOLATION, error.reason, step_id))
        except ControlError as error:
            return Stop(self._results.failure(FailureCode.SESSION_LOST, str(error), step_id))
        except SurfaceError:
            return KeepWaiting()
        return KeepWaiting()
