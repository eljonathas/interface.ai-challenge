from datetime import UTC, datetime

import pytest
from support import configs
from support.artifacts import WORKSPACE, savings_targets

from interface_ai_challenge.application.compiler import CapabilityCompiler
from interface_ai_challenge.application.discovery.recorder import RecordedAction, RecordedExtraction, Trajectory
from interface_ai_challenge.application.discovery.target_synthesis import TargetProposal
from interface_ai_challenge.domain.artifact import Provenance
from interface_ai_challenge.domain.errors import CompilationError
from interface_ai_challenge.domain.predicates import AllOf, FieldEquals, TextEquals, Visible
from interface_ai_challenge.domain.steps import ClickAction, Effect, FillAction, NavigateAction
from interface_ai_challenge.domain.values import InputRef


def _proposal(name: str, parametric: bool = False) -> TargetProposal:
    return TargetProposal(
        target=savings_targets()[name].strategies[0], rationale=f"rationale for {name}", parametric=parametric
    )


def _trajectory() -> Trajectory:
    field = _proposal("member_id_field")
    return Trajectory(
        entry_path="/desk",
        actions=[
            RecordedAction(kind="click", proposals=(field,), effect=Effect.REVERSIBLE_INPUT, summary="field"),
            RecordedAction(
                kind="fill",
                proposals=(field,),
                effect=Effect.REVERSIBLE_INPUT,
                summary="field",
                value=InputRef(name="member_id"),
            ),
            RecordedAction(
                kind="click", proposals=(_proposal("search_button"),), effect=Effect.SUBMIT, summary="Search"
            ),
            RecordedAction(
                kind="click",
                proposals=(_proposal("open_link_in_row", parametric=True),),
                effect=Effect.NAVIGATION,
                summary="Open",
            ),
            RecordedAction(
                kind="click", proposals=(_proposal("accounts_visual"),), effect=Effect.NAVIGATION, summary="canvas"
            ),
        ],
        extractions={
            name: RecordedExtraction(output=name, proposals=(_proposal(f"output_{name}"),), frame=WORKSPACE)
            for name in ("account_type", "balance", "currency")
        },
        identity_targets={"member_id": savings_targets()["identity_member_id"].strategies[0]},
    )


def _provenance(human_interventions: int = 0) -> Provenance:
    return Provenance(
        discovery_run_id="discovery-test",
        provider="openai",
        model="gpt-5.6-sol",
        compiler_version="0.1.0",
        created_at=datetime(2026, 9, 14, tzinfo=UTC),
        human_interventions=human_interventions,
    )


def test_compiles_verified_trajectory_into_checked_steps() -> None:
    artifact = CapabilityCompiler(configs.profile()).compile(configs.contract(), _trajectory(), _provenance())
    actions = [step.action for step in artifact.steps]
    assert isinstance(actions[0], NavigateAction)
    assert isinstance(actions[1], FillAction), "focus click before typing is folded into the fill"
    assert [type(action) for action in actions[2:]] == [ClickAction, ClickAction, ClickAction]
    assert artifact.steps[0].resume_boundary
    assert isinstance(artifact.steps[1].postcondition, FieldEquals)
    assert artifact.steps[2].postcondition == Visible(target_ref=artifact.steps[3].target_ref())
    assert isinstance(artifact.steps[-1].postcondition, AllOf)
    assert TextEquals(target_ref="identity_member_id", value=InputRef(name="member_id")) in artifact.success.predicates  # type: ignore[union-attr]
    assert "visual.template" in artifact.application.requires
    assert {handler.id for handler in artifact.handlers} >= {"member_not_found", "session_expired"}
    assert "M-10023" not in artifact.model_dump_json()


def test_refuses_to_compile_unverified_or_assisted_runs() -> None:
    trajectory = _trajectory()
    trajectory.problems.append("no stable, unique target could be derived for canvas-drawn control")
    del trajectory.extractions["currency"]
    with pytest.raises(CompilationError) as error:
        CapabilityCompiler(configs.profile()).compile(
            configs.contract(), trajectory, _provenance(human_interventions=1)
        )
    problems = " ".join(error.value.problems)
    assert "no stable, unique target" in problems
    assert "currency" in problems
    assert "human intervention" in problems
