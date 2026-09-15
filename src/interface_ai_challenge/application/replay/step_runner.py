from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from interface_ai_challenge.application.control import SessionControl
from interface_ai_challenge.application.guarded_executor import GuardedExecutor
from interface_ai_challenge.application.policy_guard import ActionIntent, EffectClassifier, riskiest
from interface_ai_challenge.application.predicate_evaluator import PredicateEvaluator
from interface_ai_challenge.application.replay.flow import (
    Continue,
    HumanResumed,
    KeepWaiting,
    RestartFrom,
    ResultFactory,
    StepOutcome,
    Stop,
)
from interface_ai_challenge.application.replay.handler_applier import HandlerApplier
from interface_ai_challenge.application.replay.observer import ConditionWaiter
from interface_ai_challenge.application.target_resolution import NamedTargetResolver
from interface_ai_challenge.domain.errors import ControlError, PolicyViolationError, SurfaceError
from interface_ai_challenge.domain.predicates import Predicate, Visible, target_refs
from interface_ai_challenge.domain.results import FailureCode
from interface_ai_challenge.domain.steps import ClickAction, Effect, FillAction, NavigateAction, PressKeyAction, Step
from interface_ai_challenge.domain.values import ValueResolver
from interface_ai_challenge.ports.evidence import EvidenceSink
from interface_ai_challenge.ports.surface import Candidate, Surface

Phase = Literal["precondition", "postcondition", "checkpoint"]
_RETRY_SAFE_EFFECTS = (Effect.READ, Effect.NAVIGATION, Effect.REVERSIBLE_INPUT)


@dataclass(frozen=True)
class _Unresolved(Exception):
    target_ref: str
    candidate_count: int


class StepActionPerformer:
    def __init__(
        self,
        surface: Surface,
        resolver: NamedTargetResolver,
        executor: GuardedExecutor,
        control: SessionControl,
        classifier: EffectClassifier,
        values: ValueResolver,
        results: ResultFactory,
        evidence: EvidenceSink,
    ) -> None:
        self._surface = surface
        self._resolver = resolver
        self._executor = executor
        self._control = control
        self._classifier = classifier
        self._values = values
        self._results = results
        self._evidence = evidence

    async def perform(self, step: Step) -> Continue | Stop:
        for attempt in range(1, step.max_attempts + 1):
            try:
                await self._execute(step)
                return Continue()
            except PolicyViolationError as error:
                return Stop(
                    self._results.failure(FailureCode.POLICY_VIOLATION, error.reason, step.id, {"effect": step.effect})
                )
            except ControlError as error:
                return Stop(self._results.failure(FailureCode.SESSION_LOST, str(error), step.id))
            except _Unresolved as unresolved:
                code = FailureCode.TARGET_AMBIGUOUS if unresolved.candidate_count else FailureCode.TARGET_NOT_FOUND
                return Stop(
                    self._results.failure(
                        code,
                        f"target '{unresolved.target_ref}' did not resolve to exactly one element",
                        step.id,
                        expected={"unique_target": unresolved.target_ref},
                        observed={"candidate_count": unresolved.candidate_count},
                    )
                )
            except SurfaceError as error:
                self._evidence.emit(
                    "step_action_failed", {"step_id": step.id, "attempt": attempt, "error": str(error)[:200]}
                )
                if attempt == step.max_attempts or step.effect not in _RETRY_SAFE_EFFECTS:
                    return Stop(
                        self._results.failure(
                            FailureCode.ACTION_FAILED,
                            "the UI did not accept the action",
                            step.id,
                            observed={"attempts": attempt},
                            retryable=step.effect in _RETRY_SAFE_EFFECTS,
                        )
                    )
        return Continue()

    async def _execute(self, step: Step) -> None:
        token = self._control.automation_token()
        action = step.action
        strategy_index: int | None = None
        match action:
            case NavigateAction(base_url=base_url, path=path):
                url = self._values.resolve(base_url).rstrip("/") + path
                intent = ActionIntent(command="navigate", effect=riskiest(step.effect, Effect.NAVIGATION), url=url)
                await self._executor.run(token, intent, lambda: self._surface.navigator.goto(url))
            case ClickAction(target_ref=ref):
                candidate, strategy_index = await self._unique(ref)
                role, name = self._resolver.control_identity(ref)
                effect = riskiest(step.effect, self._classifier.for_control(role, name))
                intent = ActionIntent(command="click", effect=effect, control_name=name)
                await self._executor.run(token, intent, lambda: self._surface.actions.click(candidate, step.timeout_ms))
            case FillAction(target_ref=ref, value=value):
                candidate, strategy_index = await self._unique(ref)
                text = self._values.resolve(value)
                intent = ActionIntent(command="fill", effect=Effect.REVERSIBLE_INPUT, text_length=len(text))
                await self._executor.run(
                    token, intent, lambda: self._surface.actions.fill(candidate, text, step.timeout_ms)
                )
            case PressKeyAction(target_ref=ref, key=key):
                candidate, strategy_index = await self._unique(ref)
                key_effect = Effect.SUBMIT if key == "Enter" else Effect.REVERSIBLE_INPUT
                intent = ActionIntent(command="press_key", effect=riskiest(step.effect, key_effect), keys=(key,))
                await self._executor.run(
                    token, intent, lambda: self._surface.actions.press(candidate, key, step.timeout_ms)
                )
        self._evidence.emit(
            "step_action_executed",
            {
                "step_id": step.id,
                "kind": action.kind,
                "target_ref": step.target_ref(),
                "strategy_index": strategy_index,
            },
        )

    async def _unique(self, ref: str) -> tuple[Candidate, int | None]:
        resolution = await self._resolver.resolve(ref, self._values)
        if resolution.unique is None:
            raise _Unresolved(target_ref=ref, candidate_count=len(resolution.candidates))
        return resolution.unique, resolution.strategy_index


class StepRunner:
    def __init__(
        self,
        steps: tuple[Step, ...],
        waiter: ConditionWaiter,
        evaluator: PredicateEvaluator,
        performer: StepActionPerformer,
        applier: HandlerApplier,
        surface: Surface,
        results: ResultFactory,
        evidence: EvidenceSink,
        checkpoint_timeout_ms: int,
    ) -> None:
        self._steps = steps
        self._waiter = waiter
        self._evaluator = evaluator
        self._performer = performer
        self._applier = applier
        self._surface = surface
        self._results = results
        self._evidence = evidence
        self._checkpoint_timeout_ms = checkpoint_timeout_ms

    async def run_step(self, index: int) -> StepOutcome:
        step = self._steps[index]
        self._evidence.emit(
            "step_started", {"step_id": step.id, "index": index, "effect": step.effect, "origin": step.origin}
        )
        if step.precondition is not None:
            outcome = await self.await_condition(step.precondition, step, index, "precondition")
            if not isinstance(outcome, Continue):
                return outcome
        performed = await self._performer.perform(step)
        if isinstance(performed, Stop):
            return performed
        outcome = await self.await_condition(step.postcondition, step, index, "postcondition")
        self._evidence.emit("step_finished", {"step_id": step.id, "outcome": type(outcome).__name__})
        return outcome

    async def run_checkpoint(self, success: Predicate) -> StepOutcome:
        outcome = await self.await_condition(success, None, len(self._steps), "checkpoint")
        self._evidence.emit("checkpoint_evaluated", {"outcome": type(outcome).__name__})
        return outcome

    async def await_condition(self, condition: Predicate, step: Step | None, index: int, phase: Phase) -> StepOutcome:
        timeout_ms = step.timeout_ms if step else self._checkpoint_timeout_ms
        expected: dict[str, Any] = {phase: condition.model_dump(mode="json")}
        while True:
            observation = await self._waiter.wait(condition, timeout_ms)
            if observation.conflicting:
                return Stop(
                    self._results.failure(
                        FailureCode.AMBIGUOUS_STATE,
                        f"{phase} holds while a terminal state is also visible",
                        step.id if step else None,
                        expected=expected,
                        observed={"handlers": [handler.id for handler in observation.active_handlers]},
                    )
                )
            if observation.active_handlers:
                effect = await self._applier.apply(observation.active_handlers[0], step, expected)
                if isinstance(effect, KeepWaiting):
                    timeout_ms = (step.timeout_ms if step else self._checkpoint_timeout_ms) + effect.extra_ms
                    continue
                if isinstance(effect, HumanResumed):
                    return await self._after_resume(condition, index)
                return effect
            if observation.condition_met:
                return Continue()
            observed = await self._diagnose(condition)
            if not self._applier.can_escalate:
                return Stop(self._timeout_failure(condition, step, phase, timeout_ms, observed))
            escalation = await self._applier.escalate(
                f"{phase} not satisfied within {timeout_ms} ms",
                step,
                expected={phase: condition.model_dump(mode="json")},
                observed=observed,
            )
            if isinstance(escalation, HumanResumed):
                return await self._after_resume(condition, index)
            if isinstance(escalation, Stop):
                return escalation

    async def _after_resume(self, condition: Predicate, index: int) -> StepOutcome:
        if await self._evaluator.holds(condition):
            self._evidence.emit("resume_reconciled", {"index": index, "decision": "condition_already_satisfied"})
            return Continue()
        boundary = next(
            (i for i in range(min(index, len(self._steps) - 1), -1, -1) if self._steps[i].resume_boundary),
            0,
        )
        self._evidence.emit("resume_reconciled", {"index": index, "decision": "restart", "boundary": boundary})
        return RestartFrom(index=boundary, reason="resumed after human intervention")

    async def _diagnose(self, condition: Predicate) -> dict[str, Any]:
        observed: dict[str, Any] = {"candidate_counts": {}}
        for ref in sorted(target_refs(condition)):
            observed["candidate_counts"][ref] = await self._evaluator.candidate_count(ref)
        try:
            observed["dialogs"] = await self._surface.probe.dialog_titles()
            observed["path"] = urlsplit(await self._surface.probe.current_url()).path
        except SurfaceError:
            observed["surface"] = "unavailable"
        return observed

    def _timeout_failure(
        self, condition: Predicate, step: Step | None, phase: Phase, timeout_ms: int, observed: dict[str, Any]
    ) -> Any:
        counts: dict[str, int] = observed.get("candidate_counts", {})
        if any(count > 1 for count in counts.values()):
            code = FailureCode.TARGET_AMBIGUOUS
        elif isinstance(condition, Visible) and counts.get(condition.target_ref) == 0:
            code = FailureCode.TARGET_NOT_FOUND
        elif phase == "checkpoint":
            code = FailureCode.CHECKPOINT_FAILED
        else:
            code = FailureCode.POSTCONDITION_TIMEOUT
        reason = "a target stays ambiguous" if code is FailureCode.TARGET_AMBIGUOUS else f"not within {timeout_ms} ms"
        return self._results.failure(
            code,
            f"{phase} was not satisfied ({reason}) and no known state explains it",
            step.id if step else None,
            expected={phase: condition.model_dump(mode="json")},
            observed=observed,
            retryable=code is FailureCode.POSTCONDITION_TIMEOUT,
        )
