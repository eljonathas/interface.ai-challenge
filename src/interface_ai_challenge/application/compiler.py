from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from interface_ai_challenge.application.discovery.recorder import RecordedAction, Trajectory
from interface_ai_challenge.application.discovery.target_synthesis import TargetProposal
from interface_ai_challenge.domain.artifact import (
    SCHEMA_VERSION,
    ApplicationRef,
    CapabilityArtifact,
    Entry,
    Extraction,
    Provenance,
)
from interface_ai_challenge.domain.contract import CapabilityContract, output_row_problems
from interface_ai_challenge.domain.errors import CompilationError
from interface_ai_challenge.domain.handlers import BusinessOutcomeResponse
from interface_ai_challenge.domain.predicates import AllOf, FieldEquals, Predicate, TextEquals, Visible
from interface_ai_challenge.domain.profile import TargetProfile
from interface_ai_challenge.domain.steps import (
    ClickAction,
    Effect,
    FillAction,
    NavigateAction,
    Origin,
    PressKeyAction,
    Step,
    StepAction,
)
from interface_ai_challenge.domain.targets import (
    LabeledFieldLocator,
    LabeledValueLocator,
    NamedTarget,
    RoleLocator,
    TableCellLocator,
    Target,
    VisualTarget,
)
from interface_ai_challenge.domain.values import BindingRef, InputRef

COMPILER_VERSION = "0.2.0"


class TargetCatalog:
    def __init__(self, reserved: Mapping[str, NamedTarget]) -> None:
        self._targets: dict[str, NamedTarget] = dict(reserved)

    def add(self, base_name: str, proposals: Sequence[TargetProposal]) -> str:
        strategies = tuple(proposal.target for proposal in proposals)
        for name, existing in self._targets.items():
            if existing.strategies == strategies:
                return name
        name = _unique_name(_slug(base_name), self._targets)
        rationale = " | fallback: ".join(proposal.rationale for proposal in proposals)
        self._targets[name] = NamedTarget(strategies=strategies, rationale=rationale)
        return name

    def as_dict(self) -> dict[str, NamedTarget]:
        return dict(self._targets)


class CapabilityCompiler:
    """Converts a verified discovery trajectory into a model-independent, reviewable capability."""

    def __init__(self, profile: TargetProfile) -> None:
        self._profile = profile

    def compile(
        self, contract: CapabilityContract, trajectory: Trajectory, provenance: Provenance
    ) -> CapabilityArtifact:
        self._raise_on_problems(contract, trajectory, provenance)
        # Outcome codes belong to the application's authored handlers, not to each goal.
        outcomes = {h.response.code for h in self._profile.handlers if isinstance(h.response, BusinessOutcomeResponse)}
        contract = contract.model_copy(
            update={"business_outcomes": tuple(sorted(outcomes | set(contract.business_outcomes)))}
        )
        catalog = TargetCatalog(self._profile.targets)
        actions = _without_redundant_focus_clicks(trajectory.actions)
        action_refs = [catalog.add(_base_name(action.proposals[0].target), action.proposals) for action in actions]
        extraction_refs = {
            name: catalog.add(f"output_{name}", recorded.proposals) for name, recorded in trajectory.extractions.items()
        }
        identity_refs = {
            input_name: catalog.add(
                f"identity_{input_name}",
                [TargetProposal(target=target, rationale="visible identity value that must equal the input")],
            )
            for input_name, target in trajectory.identity_targets.items()
        }
        success = AllOf(
            predicates=(
                *(TextEquals(target_ref=ref, value=InputRef(name=name)) for name, ref in identity_refs.items()),
                *(Visible(target_ref=ref) for ref in extraction_refs.values()),
            )
        )
        steps = [self._entry_step(trajectory.entry_path, action_refs[0])]
        for index, (action, ref) in enumerate(zip(actions, action_refs, strict=True)):
            following = action_refs[index + 1] if index + 1 < len(action_refs) else None
            steps.append(self._step(index + 1, action, ref, following, success))
        requires = set(self._profile.requires)
        if any(isinstance(target, VisualTarget) for action in actions for target in _targets(action)):
            requires.add("visual.template")
        return CapabilityArtifact(
            schema_version=SCHEMA_VERSION,
            capability_version="1.0.0",
            contract=contract,
            application=ApplicationRef(
                product=self._profile.product,
                ui_family=self._profile.ui_family,
                requires=tuple(sorted(requires)),  # type: ignore[arg-type]
            ),
            entry=Entry(path=trajectory.entry_path),
            targets=catalog.as_dict(),
            steps=tuple(steps),
            handlers=self._profile.handlers,
            success=success,
            extractions=tuple(Extraction(output=name, target_ref=ref) for name, ref in extraction_refs.items()),
            provenance=provenance,
        )

    def _raise_on_problems(self, contract: CapabilityContract, trajectory: Trajectory, provenance: Provenance) -> None:
        problems = list(trajectory.problems)
        problems.extend(
            output_row_problems(
                {
                    name: tuple(proposal.target for proposal in extraction.proposals)
                    for name, extraction in trajectory.extractions.items()
                }
            )
        )
        if provenance.human_interventions:
            problems.append("discovery needed human intervention; manual steps are not converted into automation")
        if not trajectory.actions:
            problems.append("discovery recorded no replayable actions")
        missing_outputs = [field.name for field in contract.outputs if field.name not in trajectory.extractions]
        if missing_outputs:
            problems.append(f"outputs without a verified extraction: {missing_outputs}")
        missing_identity = [
            field.name
            for field in contract.inputs
            if field.identifies_record and field.name not in trajectory.identity_targets
        ]
        if missing_identity:
            problems.append(f"identity checkpoints not verified for inputs: {missing_identity}")
        if problems:
            raise CompilationError(problems)

    @staticmethod
    def _entry_step(path: str, first_target: str) -> Step:
        return Step(
            id="s00_open_entry",
            action=NavigateAction(base_url=BindingRef(name="base_url"), path=path),
            postcondition=Visible(target_ref=first_target),
            effect=Effect.NAVIGATION,
            resume_boundary=True,
            max_attempts=2,
            origin=Origin.COMPILED,
        )

    @staticmethod
    def _step(index: int, action: RecordedAction, ref: str, following: str | None, success: Predicate) -> Step:
        step_action: StepAction
        postcondition: Predicate
        if action.kind == "fill":
            assert action.value is not None
            step_action = FillAction(target_ref=ref, value=action.value)
            postcondition = FieldEquals(target_ref=ref, value=action.value)
        elif action.kind == "press_key":
            assert action.key is not None
            step_action = PressKeyAction(target_ref=ref, key=action.key)
            postcondition = Visible(target_ref=following) if following else success
        else:
            step_action = ClickAction(target_ref=ref)
            postcondition = Visible(target_ref=following) if following else success
        return Step(
            id=f"s{index:02d}_{action.kind}_{ref}",
            action=step_action,
            precondition=Visible(target_ref=ref),
            postcondition=postcondition,
            effect=action.effect,
            origin=Origin.DISCOVERED,
        )


def _without_redundant_focus_clicks(actions: Sequence[RecordedAction]) -> list[RecordedAction]:
    kept: list[RecordedAction] = []
    for index, action in enumerate(actions):
        following = actions[index + 1] if index + 1 < len(actions) else None
        focuses_next_target = (
            action.kind == "click"
            and action.effect is Effect.REVERSIBLE_INPUT
            and following is not None
            and following.primary == action.primary
        )
        if not focuses_next_target:
            kept.append(action)
    return kept


def _targets(action: RecordedAction) -> list[Target]:
    return [proposal.target for proposal in action.proposals]


def _base_name(target: Target) -> str:
    if isinstance(target, VisualTarget):
        return "visual_control"
    locator = target.locator
    if isinstance(locator, RoleLocator):
        return f"{locator.name}_{locator.role}"
    if isinstance(locator, LabeledFieldLocator):
        return f"{locator.label}_field"
    if isinstance(locator, LabeledValueLocator):
        return f"{locator.label}_value"
    if isinstance(locator, TableCellLocator) and locator.control is not None:
        return f"{locator.control.name}_{locator.control.role}_in_row"
    return f"{locator.column}_cell"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_") or "target"


def _unique_name(base: str, taken: Mapping[str, object]) -> str:
    if base not in taken:
        return base
    suffix = 2
    while f"{base}_{suffix}" in taken:
        suffix += 1
    return f"{base}_{suffix}"
