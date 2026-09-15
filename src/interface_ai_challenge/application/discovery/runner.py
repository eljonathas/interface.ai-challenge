from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from interface_ai_challenge.application.commands import ComputerCommandExecutor
from interface_ai_challenge.application.control import SessionControl
from interface_ai_challenge.application.discovery.placeholders import PlaceholderError, PlaceholderResolver
from interface_ai_challenge.application.discovery.recorder import RecordedAction, Trajectory, TrajectoryRecorder
from interface_ai_challenge.application.handoff import EscalationContext, InterventionCoordinator
from interface_ai_challenge.domain.computer import ClickCommand, ComputerCommand, KeypressCommand, TypeCommand
from interface_ai_challenge.domain.contract import CapabilityContract
from interface_ai_challenge.domain.errors import ControlError, PolicyViolationError, SurfaceError
from interface_ai_challenge.domain.intervention import InterventionDecision
from interface_ai_challenge.ports.evidence import EvidenceSink
from interface_ai_challenge.ports.model import (
    AgentFeedback,
    AgentSignal,
    AgentTask,
    AgentTurn,
    ComputerUseAgent,
    FinishSignal,
    ReportIdentitySignal,
    ReportOutputSignal,
    RequestHelpSignal,
    TokenUsage,
)
from interface_ai_challenge.ports.surface import RawInput, ScreenCapture

_RESUME_BEHAVIOR = "The agent receives a fresh screenshot; pending model actions are discarded."


@dataclass(frozen=True)
class DiscoveryBudget:
    max_turns: int = 30
    max_commands: int = 60
    max_seconds: float = 300.0
    max_idle_turns: int = 3
    repeat_limit: int = 4


@dataclass(frozen=True)
class DiscoveryOutcome:
    completed: bool
    reason: str
    trajectory: Trajectory
    turns: int
    usage: TokenUsage
    human_interventions: int


@dataclass(frozen=True)
class SignalReply:
    text: str
    finished: bool = False
    help_reason: str | None = None


class DiscoverySignalHandler:
    def __init__(self, recorder: TrajectoryRecorder, contract: CapabilityContract) -> None:
        self._recorder = recorder
        self._contract = contract

    async def handle(self, signal: AgentSignal) -> SignalReply:
        match signal:
            case ReportOutputSignal(output=output, point=point):
                return SignalReply(await self._recorder.record_output(output, point))
            case ReportIdentitySignal(input=input_name, point=point):
                return SignalReply(await self._recorder.record_identity(input_name, point))
            case FinishSignal():
                return await self._finish()
            case RequestHelpSignal(reason=reason):
                return SignalReply("Escalated to a human operator.", help_reason=reason)
        raise TypeError(f"unsupported signal {signal!r}")

    async def _finish(self) -> SignalReply:
        reported = set(self._recorder.trajectory.extractions)
        missing = [field.name for field in self._contract.outputs if field.name not in reported]
        if missing:
            return SignalReply(f"Cannot finish yet: report_output was not called for {missing}.")
        problems = await self._recorder.verify_identity()
        if problems:
            return SignalReply("Cannot finish yet: " + " ".join(problems))
        return SignalReply("Completion verified independently.", finished=True)


class ProgressMonitor:
    def __init__(self, repeat_limit: int) -> None:
        self._repeat_limit = repeat_limit
        self._signature: tuple[object, ...] | None = None
        self._repeats = 0

    def is_stuck(self, commands: tuple[ComputerCommand, ...], screenshot_png: bytes) -> bool:
        if not commands:
            return False
        signature = (tuple(command.model_dump_json() for command in commands), hashlib.sha256(screenshot_png).digest())
        self._repeats = self._repeats + 1 if signature == self._signature else 1
        self._signature = signature
        return self._repeats >= self._repeat_limit

    def reset(self) -> None:
        self._signature, self._repeats = None, 0


class DiscoveryRunner:
    def __init__(
        self,
        agent: ComputerUseAgent,
        commands: ComputerCommandExecutor,
        surface_input: RawInput,
        recorder: TrajectoryRecorder,
        signals: DiscoverySignalHandler,
        control: SessionControl,
        coordinator: InterventionCoordinator,
        model_screen: ScreenCapture,
        placeholders: PlaceholderResolver,
        evidence: EvidenceSink,
        subject: str,
        budget: DiscoveryBudget | None = None,
        evidence_screen: ScreenCapture | None = None,
        clock: Callable[[], float] = time.monotonic,
        observer: Callable[[AgentTurn], None] | None = None,
    ) -> None:
        self._observer = observer
        self._evidence_screen = evidence_screen
        self._agent = agent
        self._commands = commands
        self._surface_input = surface_input
        self._recorder = recorder
        self._signals = signals
        self._control = control
        self._coordinator = coordinator
        self._model_screen = model_screen
        self._placeholders = placeholders
        self._evidence = evidence
        self._subject = subject
        self._budget = budget or DiscoveryBudget()
        self._clock = clock
        self._progress = ProgressMonitor(self._budget.repeat_limit)
        self._usage = TokenUsage()
        self._turns = 0
        self._output_names: frozenset[str] = frozenset()
        self._input_names: frozenset[str] = frozenset()

    async def run(self, task: AgentTask, entry_url: str) -> DiscoveryOutcome:
        started = self._clock()
        self._output_names = frozenset(output.name for output in task.outputs)
        self._input_names = frozenset(
            ref.name for text in task.input_placeholders if (ref := self._placeholders.input_ref(text)) is not None
        )
        self._evidence.emit(
            "discovery_started", {"goal": task.goal, "provider": self._agent.provider, "model": self._agent.model}
        )
        await self._commands.navigate(self._control.automation_token(), entry_url)
        self._recorder.record_entry(urlsplit(entry_url).path)
        turn = await self._agent.begin(task, await self._model_screen.capture())
        commands_used = idle_turns = 0
        while True:
            self._account(turn)
            if self._turns > self._budget.max_turns:
                return self._outcome(False, "step budget exhausted")
            if self._clock() - started > self._budget.max_seconds:
                return self._outcome(False, "time budget exhausted")
            feedback_notes: list[str] = []
            replies: dict[str, str] = {}
            if turn.safety_checks:
                if not await self._escalated(
                    f"provider safety check: {'; '.join(turn.safety_checks)}", evidence_reason="provider_safety_check"
                ):
                    return self._outcome(False, "provider safety check was not approved by a human")
                turn = await self._after_handoff(turn, replies, acknowledge=True)
                continue
            if turn.observation_required:
                refresh_note = (
                    "No actions or signals from the previous turn were executed. Observe the current screenshot first."
                )
                self._evidence.emit("observation_refresh", {"discarded_commands": len(turn.commands)})
                await self._attach_observation()
                turn = await self._agent.respond(
                    AgentFeedback(
                        await self._model_screen.capture(),
                        {signal.call_id: refresh_note for signal in turn.signals},
                        [refresh_note],
                    )
                )
                continue
            for command in turn.commands:
                if commands_used >= self._budget.max_commands:
                    return self._outcome(False, "step budget exhausted")
                if self._clock() - started >= self._budget.max_seconds:
                    return self._outcome(False, "time budget exhausted")
                commands_used += 1
                note = await self._perform(command)
                if note is not None:
                    feedback_notes.append(note)
                    break
            if turn.commands:
                await self._settle()
            finished = False
            handed_back = False
            for signal in turn.signals:
                reply = await self._signals.handle(signal)
                replies[signal.call_id] = reply.text
                if reply.help_reason is not None:
                    if not await self._escalated(reply.help_reason, evidence_reason="model_requested_help"):
                        return self._outcome(False, "human intervention did not resume the run")
                    replies[signal.call_id] = "A human operator intervened and returned control. Re-check the screen."
                    handed_back = True
                    break
                finished = finished or reply.finished
            if handed_back:
                turn = await self._after_handoff(turn, replies)
                continue
            if finished:
                await self._attach_observation()
                return self._outcome(True, "goal completed and verified")
            idle_turns = idle_turns + 1 if not turn.commands and not turn.signals else 0
            if idle_turns >= self._budget.max_idle_turns:
                return self._outcome(False, "agent stopped acting")
            if idle_turns:
                feedback_notes.append("Respond with a computer action, report_output, finish or request_help.")
            await self._attach_observation()
            screenshot = await self._model_screen.capture()
            if self._progress.is_stuck(turn.commands, screenshot.png):
                self._progress.reset()
                if not await self._escalated("no visible progress after repeated identical actions"):
                    return self._outcome(False, "agent was stuck")
                screenshot = await self._model_screen.capture()
            turn = await self._agent.respond(AgentFeedback(screenshot, replies, feedback_notes))

    async def _after_handoff(
        self, turn: AgentTurn, replies: dict[str, str], acknowledge: bool = False
    ) -> AgentTurn:
        note = "Human control has ended. Pending actions were discarded; decide again from the current screen."
        for signal in turn.signals:
            replies.setdefault(signal.call_id, "Not executed: " + note)
        self._progress.reset()
        await self._attach_observation()
        screenshot = await self._model_screen.capture()
        return await self._agent.respond(AgentFeedback(screenshot, replies, [note], acknowledge))

    async def _perform(self, command: ComputerCommand) -> str | None:
        try:
            pending: RecordedAction | None = None
            text: str | None = None
            match command:
                case ClickCommand(point=point):
                    pending = await self._recorder.prepare_click(point)
                case TypeCommand(text=typed):
                    text = self._placeholders.resolve(typed)
                    pending = await self._recorder.prepare_fill()
                case KeypressCommand(keys=keys):
                    pending = await self._recorder.prepare_keypress(keys)
            report = await self._commands.execute(self._control.automation_token(), command, text)
            if pending is not None:
                if pending.kind == "fill":
                    pending = await self._recorder.complete_fill(pending)
                self._recorder.commit(pending)
            self._evidence.emit(
                "agent_action_executed",
                {
                    "command": _loggable(command),
                    "effect": report.effect,
                    "recorded_as": pending.kind if pending else None,
                    "target_rationale": pending.proposals[0].rationale if pending and pending.proposals else None,
                },
            )
            return None
        except PolicyViolationError as error:
            self._evidence.emit("agent_action_blocked", {"command": _loggable(command), "reason": error.reason})
            return f"Action blocked by policy: {error.reason}. Do not retry it; use another way or call request_help."
        except PlaceholderError as error:
            return str(error)
        except (SurfaceError, ControlError) as error:
            self._evidence.emit("agent_action_failed", {"command": _loggable(command), "error": type(error).__name__})
            return "The action could not be performed on the current screen."

    async def _attach_observation(self) -> None:
        if self._evidence_screen is None:
            return
        try:
            screenshot = await self._evidence_screen.capture()
        except SurfaceError:
            self._evidence.emit("screenshot_omitted", {"reason": "capture or redaction unavailable"})
            return
        self._evidence.attach_image(f"after-turn-{self._turns:02d}", screenshot.png)

    async def _settle(self) -> None:
        try:
            await self._surface_input.settle()
        except SurfaceError:
            self._evidence.emit("surface_not_settled", {})

    async def _escalated(self, reason: str, *, evidence_reason: str | None = None) -> bool:
        if not self._coordinator.available:
            self._evidence.emit("escalation_unavailable", {"reason": evidence_reason or reason})
            return False
        decision = await self._coordinator.escalate(
            EscalationContext(
                mode="discovery",
                subject=self._subject,
                step_id=None,
                reason=reason,
                evidence_reason=evidence_reason,
                resume_behavior=_RESUME_BEHAVIOR,
            )
        )
        return decision is InterventionDecision.RESUME

    def _account(self, turn: AgentTurn) -> None:
        if self._observer is not None:
            self._observer(turn)
        self._turns += 1
        usage = turn.usage
        self._usage = TokenUsage(
            input_tokens=self._usage.input_tokens + usage.input_tokens,
            cached_input_tokens=self._usage.cached_input_tokens + usage.cached_input_tokens,
            output_tokens=self._usage.output_tokens + usage.output_tokens,
        )
        self._evidence.emit(
            "model_turn",
            {
                "turn": self._turns,
                "request_id": turn.request_id,
                "commands": [_loggable(command) for command in turn.commands],
                "signals": [
                    _loggable_signal(signal, self._output_names, self._input_names) for signal in turn.signals
                ],
                "safety_check_count": len(turn.safety_checks),
                "message_omitted": turn.message is not None,
                "reasoning_omitted": bool(turn.reasoning),
                "observation_required": turn.observation_required,
                "usage": usage.__dict__,
            },
        )

    def _outcome(self, completed: bool, reason: str) -> DiscoveryOutcome:
        outcome = DiscoveryOutcome(
            completed=completed,
            reason=reason,
            trajectory=self._recorder.trajectory,
            turns=self._turns,
            usage=self._usage,
            human_interventions=self._coordinator.interventions,
        )
        self._evidence.emit(
            "discovery_finished",
            {
                "completed": completed,
                "reason": reason,
                "turns": self._turns,
                "usage": self._usage.__dict__,
                "recorded_actions": len(outcome.trajectory.actions),
                "problems": outcome.trajectory.problems,
                "human_interventions": outcome.human_interventions,
            },
        )
        return outcome


def _loggable(command: ComputerCommand) -> dict[str, object]:
    if isinstance(command, TypeCommand):
        return {
            "kind": "type",
            "text_length": len(command.text),
            "is_placeholder": command.text.strip().startswith("{{"),
        }
    return command.model_dump(mode="json")


def _loggable_signal(
    signal: AgentSignal, output_names: frozenset[str], input_names: frozenset[str]
) -> dict[str, object]:
    """Persist signal structure, never free model text that may repeat screen data."""
    match signal:
        case ReportOutputSignal(output=output, point=point):
            return {
                "kind": "report_output",
                "output": output if output in output_names else "undeclared",
                "point": point.model_dump(),
            }
        case ReportIdentitySignal(input=input_name, point=point):
            return {
                "kind": "report_identity",
                "input": input_name if input_name in input_names else "undeclared",
                "point": point.model_dump(),
            }
        case FinishSignal():
            return {"kind": "finish"}
        case RequestHelpSignal():
            return {"kind": "request_help"}
    return {"kind": "unknown"}
