from __future__ import annotations

import secrets
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from interface_ai_challenge.adapters.console_log import ConsoleEvidenceSink, print_model_turn
from interface_ai_challenge.adapters.file_store import CapabilityFileRepository, FileAssetStore
from interface_ai_challenge.adapters.jsonl_evidence import JsonlEvidenceSink
from interface_ai_challenge.adapters.opencv_imaging import OpenCvImageEditor, OpenCvTemplateMatcher
from interface_ai_challenge.adapters.operator_console.app import OperatorConsoleServer, create_operator_app
from interface_ai_challenge.adapters.playwright.browser import open_isolated_page
from interface_ai_challenge.adapters.playwright.surface import build_playwright_surface
from interface_ai_challenge.application.commands import ComputerCommandExecutor
from interface_ai_challenge.application.compiler import COMPILER_VERSION, CapabilityCompiler
from interface_ai_challenge.application.control import SessionControl
from interface_ai_challenge.application.discovery.placeholders import PlaceholderResolver
from interface_ai_challenge.application.discovery.recorder import TrajectoryRecorder
from interface_ai_challenge.application.discovery.runner import (
    DiscoveryBudget,
    DiscoveryRunner,
    DiscoverySignalHandler,
)
from interface_ai_challenge.application.discovery.target_synthesis import (
    VisualTargetSynthesizer,
    WebTargetSynthesizer,
)
from interface_ai_challenge.application.guarded_executor import GuardedExecutor
from interface_ai_challenge.application.handoff import InterventionCoordinator
from interface_ai_challenge.application.intervention_inbox import InMemoryInterventionInbox
from interface_ai_challenge.application.operator_service import OperatorService
from interface_ai_challenge.application.policy_guard import EffectClassifier, PolicyGuard
from interface_ai_challenge.application.redaction import MaskedScreenCapture, RedactingEvidenceSink, Redactor
from interface_ai_challenge.application.replay.engine import ReplayEngine, ReplayRuntime, ReplaySettings
from interface_ai_challenge.domain.artifact import Provenance
from interface_ai_challenge.domain.contract import CapabilityContract
from interface_ai_challenge.domain.errors import ArtifactIncompatibleError, CompilationError
from interface_ai_challenge.domain.intervention import InterventionRequest
from interface_ai_challenge.domain.policy import PolicyConfig
from interface_ai_challenge.domain.profile import RedactionRules, TargetProfile
from interface_ai_challenge.domain.results import FailureCode, RunFailure, RunResult
from interface_ai_challenge.domain.values import ValueResolver
from interface_ai_challenge.ports.evidence import EvidenceSink
from interface_ai_challenge.ports.imaging import AssetStore
from interface_ai_challenge.ports.model import AgentTask, ComputerUseAgent, OutputSpec, TokenUsage
from interface_ai_challenge.ports.surface import Surface
from interface_ai_challenge.settings import RuntimeSettings

Mode = Literal["discovery", "replay"]


@dataclass(frozen=True)
class SessionServices:
    run_id: str
    run_dir: Path
    surface: Surface
    control: SessionControl
    guard: PolicyGuard
    classifier: EffectClassifier
    executor: GuardedExecutor
    commands: ComputerCommandExecutor
    redactor: Redactor
    evidence: EvidenceSink
    coordinator: InterventionCoordinator
    evidence_screen: MaskedScreenCapture
    model_screen: MaskedScreenCapture
    matcher: OpenCvTemplateMatcher
    editor: OpenCvImageEditor
    operator: OperatorService | None


def new_run_id(mode: Mode) -> str:
    return f"{mode}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"


def print_intervention(request: InterventionRequest, console_url: str | None) -> None:
    where = console_url or "no operator console is running"
    # Discovery reasons are untrusted model text. Show them only in the live
    # operator console, not on stderr where redirected logs could retain PII.
    reason = "discovery requires human intervention" if request.mode == "discovery" else request.reason
    print(
        f"\n[intervention] {reason} (step: {request.step_id or '-'})\n  take over at: {where}\n",
        file=sys.stderr,
    )


@asynccontextmanager
async def open_session(
    mode: Mode,
    run_dir: Path,
    policy: PolicyConfig,
    redaction: RedactionRules,
    runtime: RuntimeSettings,
    assets: AssetStore,
    announce: Callable[[InterventionRequest, str | None], None] = print_intervention,
) -> AsyncIterator[SessionServices]:
    """Composition root for one live session: one browser context, one control model, one evidence stream."""

    run_id = new_run_id(mode)
    redactor = Redactor()
    sink: EvidenceSink = JsonlEvidenceSink(run_dir, run_id, mode)
    if runtime.console_events:
        sink = ConsoleEvidenceSink(sink, mode)
    evidence = RedactingEvidenceSink(sink, redactor)
    guard = PolicyGuard(policy)
    classifier = EffectClassifier(policy)

    def on_blocked(url: str) -> None:
        evidence.emit("request_blocked", {"url": url})

    async with open_isolated_page(guard.allows_url, on_blocked, runtime.viewport, runtime.headless) as page:
        matcher, editor = OpenCvTemplateMatcher(), OpenCvImageEditor()
        surface = build_playwright_surface(page, runtime.viewport, matcher, assets)
        control = SessionControl()
        executor = GuardedExecutor(control, guard, evidence)
        commands = ComputerCommandExecutor(surface, executor, classifier)
        no_values = ValueResolver({}, {})
        evidence_screen = MaskedScreenCapture(
            surface.screen, surface.locator, editor, redaction.evidence_screenshots, no_values
        )
        model_screen = MaskedScreenCapture(
            surface.screen, surface.locator, editor, redaction.model_screenshots, no_values
        )
        console_url: str | None = None
        inbox = InMemoryInterventionInbox(lambda request: announce(request, console_url))
        channel = inbox if runtime.enable_operator else None
        coordinator = InterventionCoordinator(
            run_id, control, channel, evidence_screen, evidence, runtime.intervention_timeout_seconds
        )
        operator = OperatorService(control, inbox, commands, surface.screen, evidence) if channel else None
        server: OperatorConsoleServer | None = None
        if operator is not None and runtime.operator_port is not None:
            access_token = secrets.token_urlsafe(18)
            server = OperatorConsoleServer(create_operator_app(operator, access_token), runtime.operator_port)
            await server.start()
            console_url = f"http://127.0.0.1:{runtime.operator_port}/?token={access_token}"
        try:
            yield SessionServices(
                run_id=run_id,
                run_dir=run_dir,
                surface=surface,
                control=control,
                guard=guard,
                classifier=classifier,
                executor=executor,
                commands=commands,
                redactor=redactor,
                evidence=evidence,
                coordinator=coordinator,
                evidence_screen=evidence_screen,
                model_screen=model_screen,
                matcher=matcher,
                editor=editor,
                operator=operator,
            )
        finally:
            if server is not None:
                await server.stop()


@dataclass(frozen=True)
class DiscoveryReport:
    completed: bool
    reason: str
    run_id: str
    run_dir: Path
    artifact_path: Path | None
    problems: tuple[str, ...]
    turns: int
    usage: TokenUsage


async def run_discovery(
    agent: ComputerUseAgent,
    contract: CapabilityContract,
    inputs: Mapping[str, object],
    entry_url: str,
    output_dir: Path,
    policy: PolicyConfig,
    profile: TargetProfile,
    runtime: RuntimeSettings,
    budget: DiscoveryBudget | None = None,
    session_hook: Callable[[SessionServices], None] | None = None,
) -> DiscoveryReport:
    validated = contract.validate_inputs(dict(inputs))
    assets = FileAssetStore(output_dir / "assets")
    async with open_session(
        "discovery", output_dir / "discovery-run", policy, profile.redaction, runtime, assets
    ) as session:
        if session_hook is not None:
            session_hook(session)
        for field in contract.inputs:
            if field.sensitive:
                session.redactor.register(f"input.{field.name}", validated[field.name])
        values = ValueResolver(validated, {})
        recorder = TrajectoryRecorder(
            session.surface,
            WebTargetSynthesizer(),
            VisualTargetSynthesizer(session.editor, session.matcher, assets),
            session.classifier,
            contract,
            values,
        )
        runner = DiscoveryRunner(
            agent=agent,
            commands=session.commands,
            surface_input=session.surface.raw_input,
            recorder=recorder,
            signals=DiscoverySignalHandler(recorder, contract),
            control=session.control,
            coordinator=session.coordinator,
            model_screen=session.model_screen,
            placeholders=PlaceholderResolver(values),
            evidence=session.evidence,
            subject=contract.capability_id,
            budget=budget,
            evidence_screen=session.evidence_screen,
            observer=print_model_turn if runtime.console_model_decisions else None,
        )
        task = AgentTask(
            goal=contract.goal,
            outputs=tuple(OutputSpec(name=field.name, description=field.description) for field in contract.outputs),
            input_placeholders=tuple(PlaceholderResolver.placeholder_for(field.name) for field in contract.inputs),
            identity_placeholders=tuple(
                PlaceholderResolver.placeholder_for(field.name) for field in contract.inputs if field.identifies_record
            ),
        )
        outcome = await runner.run(task, entry_url)
    report = DiscoveryReport(
        completed=outcome.completed,
        reason=outcome.reason,
        run_id=session.run_id,
        run_dir=session.run_dir,
        artifact_path=None,
        problems=tuple(outcome.trajectory.problems),
        turns=outcome.turns,
        usage=outcome.usage,
    )
    if not outcome.completed:
        session.evidence.write_document("result", _report_document(report))
        return report
    provenance = Provenance(
        discovery_run_id=session.run_id,
        provider=agent.provider,
        model=agent.model,
        compiler_version=COMPILER_VERSION,
        created_at=datetime.now(UTC),
        human_interventions=outcome.human_interventions,
    )
    try:
        artifact = CapabilityCompiler(profile).compile(contract, outcome.trajectory, provenance)
    except CompilationError as error:
        report = _replace(report, completed=False, reason="compilation failed", problems=tuple(error.problems))
        session.evidence.write_document("result", _report_document(report))
        return report
    path = CapabilityFileRepository().save(artifact, output_dir)
    report = _replace(report, artifact_path=path)
    session.evidence.emit(
        "capability_compiled", {"artifact_sha256": artifact.content_sha256(), "steps": len(artifact.steps)}
    )
    session.evidence.write_document("result", _report_document(report))
    return report


async def run_replay(
    artifact_path: Path,
    inputs: Mapping[str, object],
    bindings: Mapping[str, str],
    run_dir: Path,
    policy: PolicyConfig,
    redaction: RedactionRules,
    runtime: RuntimeSettings,
    settings: ReplaySettings | None = None,
    session_hook: Callable[[SessionServices], None] | None = None,
) -> RunResult:
    """The production path an agent invokes: artifact + typed inputs in, typed result out, no model involved."""

    repository = CapabilityFileRepository()
    try:
        artifact = repository.load(artifact_path)
    except ArtifactIncompatibleError as error:
        return RunFailure(
            run_id=new_run_id("replay"),
            capability_id="unknown",
            code=FailureCode.ARTIFACT_INCOMPATIBLE,
            message=str(error),
        )
    assets = repository.assets_for(artifact_path)
    async with open_session("replay", run_dir, policy, redaction, runtime, assets) as session:
        if session_hook is not None:
            session_hook(session)
        engine = ReplayEngine(
            ReplayRuntime(
                run_id=session.run_id,
                surface=session.surface,
                control=session.control,
                executor=session.executor,
                classifier=session.classifier,
                coordinator=session.coordinator,
                evidence=session.evidence,
                redactor=session.redactor,
                evidence_screen=session.evidence_screen,
            ),
            settings,
        )
        return await engine.run(artifact, inputs, bindings)



def _replace(report: DiscoveryReport, **changes: object) -> DiscoveryReport:
    from dataclasses import replace

    return replace(report, **changes)  # type: ignore[arg-type]


def _report_document(report: DiscoveryReport) -> dict[str, object]:
    return {
        "completed": report.completed,
        "reason": report.reason,
        "run_id": report.run_id,
        "artifact": str(report.artifact_path) if report.artifact_path else None,
        "problems": list(report.problems),
        "turns": report.turns,
        "usage": report.usage.__dict__,
    }
