from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from interface_ai_challenge.application.discovery.target_synthesis import (
    TargetProposal,
    VisualTargetSynthesizer,
    WebTargetSynthesizer,
)
from interface_ai_challenge.application.policy_guard import EffectClassifier
from interface_ai_challenge.application.predicate_evaluator import normalize_text
from interface_ai_challenge.domain.contract import CapabilityContract, output_row_problems
from interface_ai_challenge.domain.errors import ExtractionError, SurfaceError
from interface_ai_challenge.domain.geometry import Point, Rect
from interface_ai_challenge.domain.steps import Effect
from interface_ai_challenge.domain.targets import FrameRef, Target, VisualTarget
from interface_ai_challenge.domain.values import InputRef, LiteralValue, ValueRef, ValueResolver
from interface_ai_challenge.ports.surface import Surface

ReplayableKey = Literal["Enter", "Tab", "Escape"]
_REPLAYABLE_KEYS: dict[tuple[str, ...], ReplayableKey] = {("Enter",): "Enter", ("Tab",): "Tab", ("Escape",): "Escape"}
_FIELD_EDITING_KEYS = frozenset({"Backspace", "Delete", "ArrowLeft", "ArrowRight", "Home", "End"})


@dataclass(frozen=True)
class RecordedAction:
    kind: Literal["click", "fill", "press_key"]
    proposals: tuple[TargetProposal, ...]
    effect: Effect
    summary: str
    value: ValueRef | None = None
    key: Literal["Enter", "Tab", "Escape"] | None = None

    @property
    def primary(self) -> Target | None:
        return self.proposals[0].target if self.proposals else None


@dataclass(frozen=True)
class RecordedExtraction:
    output: str
    proposals: tuple[TargetProposal, ...]
    frame: FrameRef


@dataclass
class Trajectory:
    entry_path: str = "/"
    actions: list[RecordedAction] = field(default_factory=list)
    extractions: dict[str, RecordedExtraction] = field(default_factory=dict)
    identity_targets: dict[str, Target] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


class TrajectoryRecorder:
    """Observes the control each model action really hit and keeps only targets verified on the live UI."""

    def __init__(
        self,
        surface: Surface,
        web: WebTargetSynthesizer,
        visual: VisualTargetSynthesizer,
        classifier: EffectClassifier,
        contract: CapabilityContract,
        values: ValueResolver,
    ) -> None:
        self._surface = surface
        self._web = web
        self._visual = visual
        self._classifier = classifier
        self._contract = contract
        self._values = values
        self.trajectory = Trajectory()

    def record_entry(self, path: str) -> None:
        self.trajectory.entry_path = path or "/"

    async def prepare_click(self, point: Point) -> RecordedAction | None:
        descriptor = await self._surface.inspector.describe_at(point)
        if descriptor is None:
            return None
        effect = self._classifier.for_click(descriptor)
        if effect is Effect.READ:
            return None
        if descriptor.is_canvas:
            proposals = await self._visual_proposals(point, descriptor.rect)
            summary = "canvas-drawn control"
        else:
            proposals = await self._verified(self._web.propose_for_control(descriptor, self._values), point)
            summary = f"{descriptor.role} '{descriptor.name}'"
        return RecordedAction(kind="click", proposals=proposals, effect=effect, summary=summary)

    async def prepare_fill(self) -> RecordedAction | None:
        focused = await self._surface.inspector.describe_focused()
        if focused is None or focused.role != "textbox":
            self.trajectory.problems.append("text was typed without a focused text field")
            return None
        proposals = await self._verified(self._web.propose_for_control(focused, self._values), focused.rect.center())
        return RecordedAction(
            kind="fill",
            proposals=proposals,
            effect=Effect.REVERSIBLE_INPUT,
            summary=f"text field '{focused.label or focused.row_label or ''}'",
        )

    async def complete_fill(self, action: RecordedAction) -> RecordedAction:
        """Record the resulting field value, not the latest keystrokes sent to it."""
        try:
            if action.primary is None:
                raise SurfaceError("field has no stable target")
            candidates = await self._surface.locator.locate(action.primary, self._values)
            if len(candidates) != 1:
                raise SurfaceError("field is not uniquely visible after editing")
            value = await self._surface.actions.read_field(candidates[0])
        except SurfaceError:
            self.trajectory.problems.append("could not verify the field value after editing")
            raise
        return replace(action, value=self._value_ref(value))

    async def prepare_keypress(self, keys: tuple[str, ...]) -> RecordedAction | None:
        if all(key in _FIELD_EDITING_KEYS for key in keys):
            return await self.prepare_fill()
        key = _REPLAYABLE_KEYS.get(keys)
        if key is None:
            self.trajectory.problems.append(f"key combination {list(keys)} cannot be compiled into a replay step")
            return None
        focused = await self._surface.inspector.describe_focused()
        if focused is None:
            self.trajectory.problems.append(f"key '{key}' was pressed without a focused control")
            return None
        proposals = await self._verified(self._web.propose_for_control(focused, self._values), focused.rect.center())
        effect = self._classifier.for_keys(keys, focused)
        return RecordedAction(
            kind="press_key", proposals=proposals, effect=effect, summary=f"key {key} on {focused.role}", key=key
        )

    def commit(self, action: RecordedAction) -> None:
        if not action.proposals:
            self.trajectory.problems.append(f"no stable, unique target could be derived for {action.summary}")
        actions = self.trajectory.actions
        if action.kind == "fill" and actions and actions[-1].kind == "fill" and actions[-1].primary == action.primary:
            actions[-1] = action
            return
        actions.append(action)

    async def record_output(self, output: str, point: Point) -> str:
        if output not in {field.name for field in self._contract.outputs}:
            return f"'{output}' is not a declared output."
        descriptor = await self._surface.inspector.describe_at(point)
        if descriptor is None:
            return "There is no element at that point."
        proposals = await self._verified(self._web.propose_for_value(descriptor, self._values), point)
        if not proposals:
            return f"Could not derive a stable locator for '{output}'. Point at the exact cell that shows the value."
        candidates = await self._surface.locator.locate(proposals[0].target, self._values)
        raw = await self._surface.actions.read_text(candidates[0])
        try:
            self._contract.output_named(output).parser.parse(raw)
        except ExtractionError as error:
            return f"The element at that point is not a valid '{output}': {error}."
        targets = {
            name: tuple(proposal.target for proposal in extraction.proposals)
            for name, extraction in self.trajectory.extractions.items()
        }
        targets[output] = tuple(proposal.target for proposal in proposals)
        problems = output_row_problems(targets)
        if problems:
            return "Output rejected: " + "; ".join(problems) + ". Report the values from the requested record."
        self.trajectory.extractions[output] = RecordedExtraction(
            output=output, proposals=proposals, frame=descriptor.frame
        )
        return f"Recorded where '{output}' is displayed ({proposals[0].rationale})."

    async def record_identity(self, input_name: str, point: Point) -> str:
        if input_name not in {item.name for item in self._contract.inputs if item.identifies_record}:
            return f"'{input_name}' is not an input that identifies the record."
        descriptor = await self._surface.inspector.describe_at(point)
        if descriptor is None:
            return "There is no element at that point."
        if descriptor.role == "textbox":
            return "Point at the value displayed by the opened record, not at a text field."
        # A row selected by the input itself always matches the input, so it cannot prove identity.
        proposals = await self._verified(
            [proposal for proposal in self._web.propose_for_value(descriptor, self._values) if not proposal.parametric],
            point,
        )
        if not proposals:
            return f"Could not derive a stable locator for '{input_name}'. Point at the value next to its label."
        if not await self._shows_input(proposals[0].target, input_name):
            return f"The value at that point does not match '{input_name}'."
        self.trajectory.identity_targets[input_name] = proposals[0].target
        return f"Recorded where the record shows '{input_name}' ({proposals[0].rationale})."

    async def verify_identity(self) -> list[str]:
        problems: list[str] = []
        for name in (input_field.name for input_field in self._contract.inputs if input_field.identifies_record):
            target = self.trajectory.identity_targets.get(name)
            if target is None:
                problems.append(f"report_identity was not called for '{name}'.")
            elif not await self._shows_input(target, name):
                problems.append(f"The screen no longer shows the requested '{name}' next to the outputs.")
        return problems

    async def _shows_input(self, target: Target, input_name: str) -> bool:
        candidates = await self._surface.locator.locate(target, self._values)
        if len(candidates) != 1:
            return False
        shown = normalize_text(await self._surface.actions.read_text(candidates[0]))
        return shown == self._values.resolve(InputRef(name=input_name))

    def _value_ref(self, typed_text: str) -> ValueRef:
        matching = self._values.input_matching(typed_text)
        if matching is not None:
            return InputRef(name=matching)
        return LiteralValue(value=typed_text)

    async def _verified(self, proposals: list[TargetProposal], point: Point) -> tuple[TargetProposal, ...]:
        verified: list[TargetProposal] = []
        for proposal in proposals:
            try:
                candidates = await self._surface.locator.locate(proposal.target, self._values)
                if len(candidates) == 1 and await self._surface.inspector.is_same_element(candidates[0], point):
                    verified.append(proposal)
            except SurfaceError:
                continue
        if any(proposal.parametric for proposal in verified):
            verified = [proposal for proposal in verified if proposal.parametric]
        return tuple(verified)

    async def _visual_proposals(self, point: Point, container: Rect) -> tuple[TargetProposal, ...]:
        screenshot = await self._surface.screen.capture()
        proposal = self._visual.propose(screenshot, point, container)
        if proposal is None or not isinstance(proposal.target, VisualTarget):
            return ()
        candidates = await self._surface.locator.locate(proposal.target, self._values)
        if len(candidates) != 1 or not candidates[0].rect.contains(point):
            return ()
        return (proposal,)
