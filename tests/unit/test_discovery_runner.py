import asyncio
from pathlib import Path

import pytest
from support import configs
from support.evidence import InMemoryEvidenceSink
from support.fake_surface import FakeApp, fake_surface

from interface_ai_challenge.adapters.file_store import FileAssetStore
from interface_ai_challenge.adapters.opencv_imaging import OpenCvImageEditor, OpenCvTemplateMatcher
from interface_ai_challenge.application.commands import ComputerCommandExecutor
from interface_ai_challenge.application.control import SessionControl
from interface_ai_challenge.application.discovery.placeholders import PlaceholderResolver
from interface_ai_challenge.application.discovery.recorder import TrajectoryRecorder
from interface_ai_challenge.application.discovery.runner import (
    DiscoveryBudget,
    DiscoveryOutcome,
    DiscoveryRunner,
    DiscoverySignalHandler,
)
from interface_ai_challenge.application.discovery.target_synthesis import VisualTargetSynthesizer, WebTargetSynthesizer
from interface_ai_challenge.application.guarded_executor import GuardedExecutor
from interface_ai_challenge.application.handoff import InterventionCoordinator
from interface_ai_challenge.application.intervention_inbox import InMemoryInterventionInbox
from interface_ai_challenge.application.policy_guard import EffectClassifier, PolicyGuard
from interface_ai_challenge.domain.computer import TypeCommand, UnsupportedCommand, WaitCommand
from interface_ai_challenge.domain.geometry import Screenshot
from interface_ai_challenge.domain.intervention import InterventionDecision
from interface_ai_challenge.domain.values import ValueResolver
from interface_ai_challenge.ports.model import AgentFeedback, AgentTask, AgentTurn, FinishSignal, RequestHelpSignal

TASK = AgentTask(goal="goal", outputs=(), input_placeholders=("{{inputs.member_id}}",))


class ScriptedAgent:
    """Test double for the model port. Never used as evidence of a real discovery."""

    provider = "scripted"
    model = "none"

    def __init__(self, turns: list[AgentTurn]) -> None:
        self._turns = turns
        self.feedback: list[AgentFeedback] = []

    async def begin(self, task: AgentTask, screenshot: Screenshot) -> AgentTurn:
        return self._next()

    async def respond(self, feedback: AgentFeedback) -> AgentTurn:
        self.feedback.append(feedback)
        return self._next()

    def _next(self) -> AgentTurn:
        return self._turns.pop(0) if self._turns else AgentTurn(commands=(), signals=())


async def _discover(
    agent: ScriptedAgent, tmp_path: Path, budget: DiscoveryBudget, with_operator: bool = False
) -> tuple[DiscoveryOutcome, InMemoryEvidenceSink]:
    app = FakeApp()
    surface = fake_surface(app)
    evidence = InMemoryEvidenceSink()
    policy = configs.policy()
    control = SessionControl()
    inbox = InMemoryInterventionInbox()

    async def operate() -> None:
        while (request := inbox.current()) is None:
            await asyncio.sleep(0.001)
        await control.claim("test-operator")
        app.field = "changed by human"
        await control.release_to_automation("test-operator")
        inbox.decide(request.id, InterventionDecision.RESUME)

    classifier = EffectClassifier(policy)
    executor = GuardedExecutor(control, PolicyGuard(policy), evidence)
    values = ValueResolver({"member_id": "M-10023"}, {})
    contract = configs.contract()
    recorder = TrajectoryRecorder(
        surface,
        WebTargetSynthesizer(),
        VisualTargetSynthesizer(OpenCvImageEditor(), OpenCvTemplateMatcher(), FileAssetStore(tmp_path)),
        classifier,
        contract,
        values,
    )
    runner = DiscoveryRunner(
        agent=agent,
        commands=ComputerCommandExecutor(surface, executor, classifier),
        surface_input=surface.raw_input,
        recorder=recorder,
        signals=DiscoverySignalHandler(recorder, contract),
        control=control,
        coordinator=InterventionCoordinator(
            "discovery-test", control, inbox if with_operator else None, surface.screen, evidence, 1
        ),
        model_screen=surface.screen,
        placeholders=PlaceholderResolver(values),
        evidence=evidence,
        subject=contract.capability_id,
        budget=budget,
    )
    operator_task = asyncio.create_task(operate()) if with_operator else None
    try:
        return await runner.run(TASK, "http://127.0.0.1:8000/desk"), evidence
    finally:
        if operator_task is not None:
            operator_task.cancel()
            await asyncio.gather(operator_task, return_exceptions=True)


async def test_policy_blocks_are_fed_back_to_the_model(tmp_path: Path) -> None:
    agent = ScriptedAgent([AgentTurn(commands=(UnsupportedCommand(original_kind="drag"),), signals=())])
    outcome, evidence = await _discover(agent, tmp_path, DiscoveryBudget(max_idle_turns=2))
    assert "blocked by policy" in agent.feedback[0].notes[0]
    assert evidence.named("agent_action_blocked")
    assert not outcome.completed and outcome.reason == "agent stopped acting"


async def test_finish_is_verified_independently_of_the_model(tmp_path: Path) -> None:
    agent = ScriptedAgent([AgentTurn(commands=(), signals=(FinishSignal(call_id="c1", summary="done"),))])
    outcome, _ = await _discover(agent, tmp_path, DiscoveryBudget(max_idle_turns=2))
    assert "Cannot finish yet" in agent.feedback[0].signal_replies["c1"]
    assert not outcome.completed


async def test_unknown_placeholder_is_rejected_and_real_values_are_not_logged(tmp_path: Path) -> None:
    agent = ScriptedAgent([AgentTurn(commands=(TypeCommand(text="{{inputs.pin}}"),), signals=())])
    _, evidence = await _discover(agent, tmp_path, DiscoveryBudget(max_idle_turns=2))
    assert "Unknown input placeholder" in agent.feedback[0].notes[0]
    assert "M-10023" not in evidence.serialized()


async def test_help_request_without_operator_stops_the_run(tmp_path: Path) -> None:
    agent = ScriptedAgent([AgentTurn(commands=(), signals=(RequestHelpSignal(call_id="c1", reason="login"),))])
    outcome, evidence = await _discover(agent, tmp_path, DiscoveryBudget())
    assert not outcome.completed
    assert evidence.named("escalation_unavailable")


async def test_step_budget_is_enforced(tmp_path: Path) -> None:
    agent = ScriptedAgent([AgentTurn(commands=(WaitCommand(),), signals=()) for _ in range(10)])
    outcome, _ = await _discover(agent, tmp_path, DiscoveryBudget(max_turns=3, repeat_limit=99))
    assert not outcome.completed and outcome.reason == "step budget exhausted"
    assert outcome.turns == 4


@pytest.mark.parametrize("batch_sizes", [(5,), (1, 4)])
async def test_command_budget_is_enforced_before_each_action(tmp_path: Path, batch_sizes: tuple[int, ...]) -> None:
    agent = ScriptedAgent(
        [AgentTurn(commands=tuple(WaitCommand() for _ in range(size)), signals=()) for size in batch_sizes]
    )
    outcome, evidence = await _discover(agent, tmp_path, DiscoveryBudget(max_commands=1))
    assert not outcome.completed and outcome.reason == "step budget exhausted"
    assert len(evidence.named("agent_action_executed")) == 1


async def test_provider_handoff_discards_commands_and_signals(tmp_path: Path) -> None:
    agent = ScriptedAgent(
        [
            AgentTurn(
                commands=(WaitCommand(),),
                signals=(FinishSignal(call_id="finish", summary="done"),),
                safety_checks=("operator must inspect the page",),
            ),
            AgentTurn(commands=(WaitCommand(),), signals=()),
        ]
    )
    outcome, evidence = await _discover(agent, tmp_path, DiscoveryBudget(max_idle_turns=1), with_operator=True)
    assert outcome.human_interventions == 1
    assert len(evidence.named("agent_action_executed")) == 1  # Only the fresh decision executes.
    assert agent.feedback[0].screenshot.revision > 1
    assert agent.feedback[0].acknowledge_safety_checks
    assert agent.feedback[0].signal_replies["finish"].startswith("Not executed:")


async def test_help_handoff_discards_remaining_signals(tmp_path: Path) -> None:
    agent = ScriptedAgent(
        [
            AgentTurn(
                commands=(),
                signals=(
                    RequestHelpSignal(call_id="help", reason="login"),
                    FinishSignal(call_id="finish", summary="done"),
                ),
            ),
        ]
    )
    outcome, _ = await _discover(agent, tmp_path, DiscoveryBudget(max_idle_turns=1), with_operator=True)
    assert outcome.human_interventions == 1 and not outcome.completed
    assert agent.feedback[0].signal_replies["finish"].startswith("Not executed:")
    assert agent.feedback[0].screenshot.revision > 1


async def test_observation_refresh_never_executes_stale_actions_or_signals(tmp_path: Path) -> None:
    agent = ScriptedAgent(
        [
            AgentTurn(
                commands=(TypeCommand(text="must not be typed"),),
                signals=(RequestHelpSignal(call_id="stale-help", reason="must not escalate"),),
                observation_required=True,
            ),
            AgentTurn(commands=(WaitCommand(),), signals=()),
        ]
    )
    _, evidence = await _discover(agent, tmp_path, DiscoveryBudget(max_idle_turns=1))
    assert evidence.named("observation_refresh") == [{"discarded_commands": 1}]
    assert not evidence.named("escalation_unavailable")
    assert len(evidence.named("agent_action_executed")) == 1
    assert evidence.named("agent_action_executed")[0]["command"]["kind"] == "wait"
    assert agent.feedback[0].screenshot.revision > 1
    assert "No actions or signals" in agent.feedback[0].signal_replies["stale-help"]


@pytest.mark.parametrize("with_operator", [False, True])
@pytest.mark.parametrize("signal_kind", ["finish", "help", "safety"])
async def test_model_screen_data_is_not_persisted_in_any_discovery_log(
    tmp_path: Path, with_operator: bool, signal_kind: str
) -> None:
    # These values are not registered with a redactor. The model may repeat
    # anything it sees, so free text must never reach the evidence sink.
    private_text = "Alice Carter has a savings balance of $1,250.45"
    signal = (
        RequestHelpSignal(call_id="help", reason=private_text)
        if signal_kind == "help"
        else FinishSignal(call_id="finish", summary=private_text)
    )
    agent = ScriptedAgent(
        [
            AgentTurn(
                commands=(),
                signals=(signal,),
                message=private_text,
                safety_checks=(private_text,) if signal_kind == "safety" else (),
            )
        ]
    )
    _, evidence = await _discover(agent, tmp_path, DiscoveryBudget(max_idle_turns=1), with_operator=with_operator)
    assert "Alice Carter" not in evidence.serialized()
    assert "1,250.45" not in evidence.serialized()
    assert evidence.named("model_turn")[0]["signals"][0]["kind"] in {"finish", "request_help"}
    if signal_kind in {"help", "safety"}:
        event = "intervention_requested" if with_operator else "escalation_unavailable"
        assert evidence.named(event)[0]["reason"] == (
            "model_requested_help" if signal_kind == "help" else "provider_safety_check"
        )
