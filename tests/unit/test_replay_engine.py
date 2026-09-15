import asyncio
import time
from dataclasses import dataclass

import pytest
from support import configs
from support.artifacts import savings_artifact
from support.evidence import InMemoryEvidenceSink
from support.fake_surface import FakeApp, fake_surface

from interface_ai_challenge.application.commands import ComputerCommandExecutor
from interface_ai_challenge.application.control import SessionControl
from interface_ai_challenge.application.guarded_executor import GuardedExecutor
from interface_ai_challenge.application.handoff import InterventionCoordinator
from interface_ai_challenge.application.intervention_inbox import InMemoryInterventionInbox
from interface_ai_challenge.application.operator_service import OperatorService
from interface_ai_challenge.application.policy_guard import EffectClassifier, PolicyGuard
from interface_ai_challenge.application.redaction import RedactingEvidenceSink, Redactor
from interface_ai_challenge.application.replay.engine import ReplayEngine, ReplayRuntime, ReplaySettings
from interface_ai_challenge.domain.artifact import CapabilityArtifact
from interface_ai_challenge.domain.computer import ClickCommand
from interface_ai_challenge.domain.geometry import Point
from interface_ai_challenge.domain.results import RunResult

BINDINGS = {"base_url": "http://127.0.0.1:8000"}


@dataclass
class Harness:
    app: FakeApp
    evidence: InMemoryEvidenceSink
    engine: ReplayEngine
    operator: OperatorService
    inbox: InMemoryInterventionInbox

    async def run(self, member_id: str, artifact: CapabilityArtifact | None = None, **bindings: str) -> RunResult:
        return await self.engine.run(artifact or savings_artifact(), {"member_id": member_id}, bindings or BINDINGS)


def harness(scenario: str = "normal", with_operator: bool = False) -> Harness:
    app = FakeApp(scenario)
    surface = fake_surface(app)
    raw_evidence = InMemoryEvidenceSink()
    redactor = Redactor()
    evidence = RedactingEvidenceSink(raw_evidence, redactor)
    policy = configs.policy()
    control = SessionControl()
    guard, classifier = PolicyGuard(policy), EffectClassifier(policy)
    executor = GuardedExecutor(control, guard, evidence)
    inbox = InMemoryInterventionInbox()
    coordinator = InterventionCoordinator(
        "replay-test", control, inbox if with_operator else None, surface.screen, evidence, timeout_seconds=2
    )
    runtime = ReplayRuntime(
        "replay-test", surface, control, executor, classifier, coordinator, evidence, redactor, surface.screen
    )
    engine = ReplayEngine(
        runtime, ReplaySettings(checkpoint_timeout_ms=300, poll_seconds=0.01, ambiguity_grace_seconds=0.05)
    )
    operator = OperatorService(
        control, inbox, ComputerCommandExecutor(surface, executor, classifier), surface.screen, evidence
    )
    return Harness(app, raw_evidence, engine, operator, inbox)


async def test_success_returns_typed_outputs_for_different_parameters() -> None:
    first = await harness().run("M-10023")
    second = await harness().run("M-20417")
    assert first.status == "success" and first.outputs == {
        "account_type": "savings",
        "balance": "1250.45",
        "currency": "USD",
    }
    assert second.status == "success" and second.outputs["balance"] == "12980.07"
    assert not first.assisted


async def test_prefix_matches_do_not_confuse_row_targeting() -> None:
    result = await harness().run("M-10023")
    assert result.status == "success"


async def test_sensitive_inputs_and_outputs_never_reach_evidence() -> None:
    run = harness()
    result = await run.run("M-10023")
    assert result.status == "success"
    persisted = run.evidence.serialized()
    assert "M-10023" not in persisted
    assert "1250.45" not in persisted
    assert "[REDACTED:output.balance]" in persisted
    assert run.evidence.named("replay_started")[0]["llm_calls"] == 0


@pytest.mark.parametrize(
    ("member_id", "code", "step_id"),
    [
        ("M-99999", "member_not_found", "s02_click_search_button"),
        ("10023", "validation_rejected", "s02_click_search_button"),
        ("M-30555", "account_not_found", "s04_click_accounts_visual"),
    ],
)
async def test_business_outcomes_are_results_not_crashes(member_id: str, code: str, step_id: str) -> None:
    result = await harness().run(member_id)
    assert result.status == "business_outcome"
    assert (result.code, result.step_id) == (code, step_id)


@pytest.mark.parametrize("scenario", ["maintenance", "transient_error"])
async def test_known_interruptions_are_recovered(scenario: str) -> None:
    run = harness(scenario)
    result = await run.run("M-20417")
    assert result.status == "success"
    triggered = {event["handler_id"] for event in run.evidence.named("handler_triggered")}
    assert triggered == {"maintenance_notice" if scenario == "maintenance" else "transient_read_error"}


async def test_permission_denied_is_a_hard_failure_with_diagnostics() -> None:
    result = await harness("permission_denied").run("M-10023")
    assert result.status == "failure"
    assert result.code == "permission_denied"
    assert result.step_id == "s04_click_accounts_visual"
    assert "postcondition" in result.expected and result.observed == {"handler": "permission_denied"}
    assert result.evidence_ref is not None


async def test_input_violations_fail_before_touching_the_ui() -> None:
    run = harness()
    result = await run.run("M-" + "1" * 40)
    assert result.status == "failure" and result.code == "input_invalid"
    assert run.app.page == "blank"


async def test_missing_binding_is_an_incompatibility() -> None:
    run = harness()
    result = await run.run("M-10023", tenant="x")
    assert result.status == "failure" and result.code == "artifact_incompatible"
    assert run.app.page == "blank"


async def test_ambiguous_visual_target_stops_before_clicking_without_waiting_for_the_step_timeout() -> None:
    run = harness("duplicate_visual")
    started = time.monotonic()
    result = await run.run("M-10023", artifact=savings_artifact(timeout_ms=10_000))
    assert result.status == "failure" and result.code == "target_ambiguous"
    assert time.monotonic() - started < 3
    assert "visual:accounts" not in run.app.clicks


async def test_conflicting_terminal_states_are_reported_as_ambiguous() -> None:
    result = await harness("conflict").run("M-10023")
    assert result.status == "failure" and result.code == "ambiguous_state"


async def test_policy_blocks_irreversible_control_even_if_artifact_references_it() -> None:
    run = harness("irreversible")
    result = await run.run("M-10023", artifact=savings_artifact(search_button_name="Close account"))
    assert result.status == "failure" and result.code == "policy_violation"
    assert "button:Close account" not in run.app.clicks


async def test_unknown_dialog_without_operator_channel_fails_explicitly() -> None:
    result = await harness("unknown_dialog").run("M-10023")
    assert result.status == "failure" and result.code == "escalation_unavailable"
    assert result.expected and result.observed == {"handler": "unexpected_dialog"}


async def _operate(run: Harness, hand_back: bool) -> None:
    while run.inbox.current() is None:
        await asyncio.sleep(0.01)
    token = await run.operator.claim("operator-1")
    screenshot = await run.operator.screenshot()
    await run.operator.act("operator-1", token.epoch, screenshot.revision, ClickCommand(point=Point(x=500, y=300)))
    if hand_back:
        await run.operator.hand_back("operator-1")
    else:
        await run.operator.abort("operator-1")


async def test_human_takes_over_same_session_and_automation_resumes_safely() -> None:
    run = harness("session_expired", with_operator=True)
    operator = asyncio.create_task(_operate(run, hand_back=True))
    result = await run.run("M-10023")
    await operator
    assert result.status == "success" and result.assisted
    events = [name for name, _ in run.evidence.events]
    for expected in (
        "intervention_requested",
        "control_claimed",
        "human_action",
        "control_returned",
        "resume_reconciled",
    ):
        assert expected in events
    assert run.evidence.named("resume_reconciled")[0]["decision"] == "restart"
    assert run.evidence.named("human_action")[0]["operator_id"] == "operator-1"


async def test_operator_abort_ends_run_and_revokes_control() -> None:
    run = harness("unknown_dialog", with_operator=True)
    operator = asyncio.create_task(_operate(run, hand_back=False))
    result = await run.run("M-10023")
    await operator
    assert result.status == "failure" and result.code == "intervention_aborted"
