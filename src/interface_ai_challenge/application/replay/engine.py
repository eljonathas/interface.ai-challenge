from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from interface_ai_challenge.application.control import SessionControl
from interface_ai_challenge.application.guarded_executor import GuardedExecutor
from interface_ai_challenge.application.handoff import InterventionCoordinator
from interface_ai_challenge.application.policy_guard import EffectClassifier
from interface_ai_challenge.application.predicate_evaluator import PredicateEvaluator
from interface_ai_challenge.application.redaction import Redactor
from interface_ai_challenge.application.replay.extraction import OutputExtractor
from interface_ai_challenge.application.replay.flow import Continue, RestartFrom, ResultFactory, Stop
from interface_ai_challenge.application.replay.handler_applier import HandlerApplier
from interface_ai_challenge.application.replay.observer import ConditionWaiter, StateObserver
from interface_ai_challenge.application.replay.step_runner import StepActionPerformer, StepRunner
from interface_ai_challenge.application.target_resolution import NamedTargetResolver
from interface_ai_challenge.domain.artifact import CapabilityArtifact
from interface_ai_challenge.domain.errors import ExtractionError, InputValidationError, SurfaceError
from interface_ai_challenge.domain.results import FailureCode, RunFailure, RunResult
from interface_ai_challenge.domain.values import ValueResolver
from interface_ai_challenge.ports.evidence import EvidenceSink
from interface_ai_challenge.ports.surface import ScreenCapture, Surface


@dataclass(frozen=True)
class ReplaySettings:
    checkpoint_timeout_ms: int = 10_000
    max_restarts: int = 2
    poll_seconds: float = 0.2
    ambiguity_grace_seconds: float = 1.0


@dataclass(frozen=True)
class ReplayRuntime:
    run_id: str
    surface: Surface
    control: SessionControl
    executor: GuardedExecutor
    classifier: EffectClassifier
    coordinator: InterventionCoordinator
    evidence: EvidenceSink
    redactor: Redactor
    evidence_screen: ScreenCapture


class ReplayEngine:
    """Interprets a capability artifact against a live surface. It has no dependency on any model client."""

    def __init__(self, runtime: ReplayRuntime, settings: ReplaySettings | None = None) -> None:
        self._runtime = runtime
        self._settings = settings or ReplaySettings()

    async def run(
        self, artifact: CapabilityArtifact, inputs: Mapping[str, object], bindings: Mapping[str, str]
    ) -> RunResult:
        runtime = self._runtime
        results = ResultFactory(runtime.run_id, artifact.capability_id, lambda: runtime.coordinator.interventions > 0)
        runtime.evidence.emit(
            "replay_started",
            {
                "capability_id": artifact.capability_id,
                "capability_version": artifact.capability_version,
                "schema_version": artifact.schema_version,
                "artifact_sha256": artifact.content_sha256(),
                "input_names": sorted(inputs),
                "llm_calls": 0,
            },
        )
        result: RunResult = await self._execute(artifact, inputs, bindings, results)
        if isinstance(result, RunFailure):
            result = result.model_copy(update={"evidence_ref": await self._failure_screenshot()})
        runtime.evidence.emit("replay_finished", {"status": result.status, "code": getattr(result, "code", None)})
        runtime.evidence.write_document("result", result.model_dump(mode="json"))
        return result

    async def _execute(
        self,
        artifact: CapabilityArtifact,
        inputs: Mapping[str, object],
        bindings: Mapping[str, str],
        results: ResultFactory,
    ) -> RunResult:
        try:
            validated = artifact.contract.validate_inputs(dict(inputs))
        except InputValidationError as error:
            return results.failure(FailureCode.INPUT_INVALID, str(error), expected={"field": error.field})
        incompatibility = self._incompatibility(artifact, bindings)
        if incompatibility:
            return results.failure(FailureCode.ARTIFACT_INCOMPATIBLE, incompatibility)
        for field in artifact.contract.inputs:
            if field.sensitive:
                self._runtime.redactor.register(f"input.{field.name}", validated[field.name])
        values = ValueResolver(validated, bindings)
        runner, extractor = self._assemble(artifact, values, results)
        index = restarts = 0
        while True:
            at_checkpoint = index == len(artifact.steps)
            outcome = await (runner.run_checkpoint(artifact.success) if at_checkpoint else runner.run_step(index))
            match outcome:
                case Stop(result=stopped):
                    return stopped
                case RestartFrom(index=boundary):
                    restarts += 1
                    if restarts > self._settings.max_restarts:
                        return results.failure(FailureCode.RECOVERY_EXHAUSTED, "too many restarts from safe boundaries")
                    index = boundary
                case Continue() if at_checkpoint:
                    break
                case Continue():
                    index += 1
        try:
            outputs = await extractor.extract(artifact.contract, artifact.extractions)
        except ExtractionError as error:
            return results.failure(FailureCode.EXTRACTION_FAILED, str(error))
        for output_field in artifact.contract.outputs:
            if output_field.sensitive:
                self._runtime.redactor.register(f"output.{output_field.name}", outputs[output_field.name])
        return results.success(outputs)

    def _incompatibility(self, artifact: CapabilityArtifact, bindings: Mapping[str, str]) -> str | None:
        missing = sorted(set(artifact.entry.required_bindings) - set(bindings))
        if missing:
            return f"missing bindings {missing}"
        unsupported = sorted(set(artifact.application.requires) - self._runtime.surface.features)
        if unsupported:
            return f"surface does not support required features {unsupported}"
        return None

    def _assemble(
        self, artifact: CapabilityArtifact, values: ValueResolver, results: ResultFactory
    ) -> tuple[StepRunner, OutputExtractor]:
        runtime = self._runtime
        surface = runtime.surface
        resolver = NamedTargetResolver(surface.locator, artifact.targets)
        evaluator = PredicateEvaluator(resolver, surface.actions, surface.probe, values)
        waiter = ConditionWaiter(
            StateObserver(evaluator, artifact.handlers),
            poll_seconds=self._settings.poll_seconds,
            ambiguity_grace_seconds=self._settings.ambiguity_grace_seconds,
        )
        applier = HandlerApplier(
            artifact.capability_id,
            resolver,
            surface.actions,
            runtime.executor,
            runtime.control,
            runtime.classifier,
            runtime.coordinator,
            values,
            results,
            runtime.evidence,
        )
        performer = StepActionPerformer(
            surface, resolver, runtime.executor, runtime.control, runtime.classifier, values, results, runtime.evidence
        )
        runner = StepRunner(
            artifact.steps,
            waiter,
            evaluator,
            performer,
            applier,
            surface,
            results,
            runtime.evidence,
            self._settings.checkpoint_timeout_ms,
        )
        return runner, OutputExtractor(resolver, surface.actions, values)

    async def _failure_screenshot(self) -> str | None:
        try:
            screenshot = await self._runtime.evidence_screen.capture()
        except SurfaceError:
            self._runtime.evidence.emit("screenshot_omitted", {"reason": "capture or redaction unavailable"})
            return None
        return self._runtime.evidence.attach_image("failure", screenshot.png)
