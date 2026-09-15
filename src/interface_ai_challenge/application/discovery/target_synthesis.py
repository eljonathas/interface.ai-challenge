from __future__ import annotations

from dataclasses import dataclass

from interface_ai_challenge.domain.elements import ElementDescriptor
from interface_ai_challenge.domain.geometry import Point, Rect, Screenshot, Size
from interface_ai_challenge.domain.targets import (
    LabeledFieldLocator,
    LabeledValueLocator,
    RoleLocator,
    TableCellLocator,
    Target,
    VisualTarget,
    WebTarget,
)
from interface_ai_challenge.domain.values import InputRef, LiteralValue, ValueRef, ValueResolver
from interface_ai_challenge.ports.imaging import AssetStore, ImageEditor, TemplateMatcher

_TEMPLATE_SIZE = Size(width=96, height=30)


@dataclass(frozen=True)
class TargetProposal:
    target: Target
    rationale: str
    parametric: bool = False


@dataclass(frozen=True)
class _RowKey:
    column: str
    value: ValueRef
    parametric: bool


class WebTargetSynthesizer:
    """Turns an observed element into candidate locators ordered from most to least robust."""

    def propose_for_control(self, descriptor: ElementDescriptor, values: ValueResolver) -> list[TargetProposal]:
        if not descriptor.frame_is_named:
            return []
        if descriptor.role == "textbox":
            return self._textbox(descriptor)
        if descriptor.role in ("button", "link"):
            return self._button_or_link(descriptor, values)
        return []

    def propose_for_value(self, descriptor: ElementDescriptor, values: ValueResolver) -> list[TargetProposal]:
        if not descriptor.frame_is_named:
            return []
        column = descriptor.column_header
        proposals = [
            TargetProposal(
                target=WebTarget(
                    frame=descriptor.frame,
                    locator=TableCellLocator(column=column, row_key_column=key.column, row_key=key.value),
                ),
                rationale=f"cell under column '{column}' in the row where '{key.column}' "
                f"{'equals the input' if key.parametric else 'has a stable label'}",
                parametric=key.parametric,
            )
            for key in (self._row_keys(descriptor, values) if column else [])
            if column
        ]
        if descriptor.row_label:
            proposals.append(
                TargetProposal(
                    target=WebTarget(frame=descriptor.frame, locator=LabeledValueLocator(label=descriptor.row_label)),
                    rationale=f"value cell next to the visible label '{descriptor.row_label}'",
                )
            )
        return proposals

    def _textbox(self, descriptor: ElementDescriptor) -> list[TargetProposal]:
        label = descriptor.label or descriptor.row_label
        if not label:
            return []
        return [
            TargetProposal(
                target=WebTarget(frame=descriptor.frame, locator=LabeledFieldLocator(label=label)),
                rationale=f"text field identified by its visible label '{label}'; generated names and ids are ignored",
            )
        ]

    def _button_or_link(self, descriptor: ElementDescriptor, values: ValueResolver) -> list[TargetProposal]:
        control = RoleLocator(role=descriptor.role, name=descriptor.name)  # type: ignore[arg-type]
        proposals: list[TargetProposal] = []
        if descriptor.column_header is not None and descriptor.row_cells:
            for key in self._row_keys(descriptor, values):
                proposals.append(
                    TargetProposal(
                        target=WebTarget(
                            frame=descriptor.frame,
                            locator=TableCellLocator(
                                column=descriptor.column_header,
                                row_key_column=key.column,
                                row_key=key.value,
                                control=control,
                            ),
                        ),
                        rationale=f"{descriptor.role} '{descriptor.name}' inside the row where '{key.column}' "
                        f"{'equals the input' if key.parametric else 'has a stable label'}",
                        parametric=key.parametric,
                    )
                )
        proposals.append(
            TargetProposal(
                target=WebTarget(frame=descriptor.frame, locator=control),
                rationale=f"{descriptor.role} identified by its exact accessible name '{descriptor.name}'",
            )
        )
        return proposals

    @staticmethod
    def _row_keys(descriptor: ElementDescriptor, values: ValueResolver) -> list[_RowKey]:
        parametric = [
            _RowKey(column=column, value=InputRef(name=name), parametric=True)
            for column, text in descriptor.row_cells.items()
            if column and (name := values.input_matching(text))
        ]
        if parametric:
            return parametric
        return [
            _RowKey(column=column, value=LiteralValue(value=text), parametric=False)
            for column, text in descriptor.row_cells.items()
            if column and text and not any(character.isdigit() for character in text)
        ]


class VisualTargetSynthesizer:
    """Builds a template anchor around a click on a surface without an addressable control (e.g. canvas)."""

    def __init__(
        self,
        editor: ImageEditor,
        matcher: TemplateMatcher,
        assets: AssetStore,
        template_size: Size = _TEMPLATE_SIZE,
        min_score: float = 0.92,
    ) -> None:
        self._editor = editor
        self._matcher = matcher
        self._assets = assets
        self._template_size = template_size
        self._min_score = min_score

    def propose(self, screenshot: Screenshot, point: Point, container: Rect) -> TargetProposal | None:
        crop = self._crop_rect(point, container, screenshot.size)
        if crop is None:
            return None
        template = self._editor.crop(screenshot.png, crop)
        if len(self._matcher.find(template, screenshot.png, None, self._min_score)) != 1:
            return None
        sha256 = self._assets.put(template)
        target = VisualTarget(
            template_sha256=sha256,
            template_size=Size(width=int(crop.width), height=int(crop.height)),
            click_offset=Point(x=point.x - int(crop.x), y=point.y - int(crop.y)),
            viewport=screenshot.size,
            min_score=self._min_score,
        )
        return TargetProposal(
            target=target,
            rationale="control drawn without a DOM element; anchored by a static image template that must match "
            "exactly once in the current screenshot",
        )

    def _crop_rect(self, point: Point, container: Rect, screen: Size) -> Rect | None:
        width, height = self._template_size.width, self._template_size.height
        left = _clamp(point.x - width / 2, container.x, min(container.x + container.width, screen.width) - width)
        top = _clamp(point.y - height / 2, container.y, min(container.y + container.height, screen.height) - height)
        rect = Rect(x=int(max(left, 0)), y=int(max(top, 0)), width=width, height=height)
        return rect if rect.contains(point) else None


def _clamp(value: float, lowest: float, highest: float) -> float:
    return min(max(value, lowest), highest)
