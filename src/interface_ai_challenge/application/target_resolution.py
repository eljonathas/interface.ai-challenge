from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from interface_ai_challenge.domain.targets import (
    LabeledFieldLocator,
    NamedTarget,
    RoleLocator,
    TableCellLocator,
    VisualTarget,
    WebTarget,
)
from interface_ai_challenge.domain.values import ValueResolver
from interface_ai_challenge.ports.surface import Candidate, TargetLocator


@dataclass(frozen=True)
class TargetResolution:
    candidates: tuple[Candidate, ...]
    strategy_index: int | None

    @property
    def unique(self) -> Candidate | None:
        return self.candidates[0] if len(self.candidates) == 1 else None


class NamedTargetResolver:
    """Tries strategies in order, moving on only when a strategy matches nothing; ambiguity is never skipped."""

    def __init__(self, locator: TargetLocator, targets: Mapping[str, NamedTarget]) -> None:
        self._locator = locator
        self._targets = targets

    async def resolve(self, target_ref: str, values: ValueResolver) -> TargetResolution:
        for index, strategy in enumerate(self._targets[target_ref].strategies):
            candidates = await self._locator.locate(strategy, values)
            if candidates:
                return TargetResolution(candidates=candidates, strategy_index=index)
        return TargetResolution(candidates=(), strategy_index=None)

    def control_identity(self, target_ref: str) -> tuple[str, str | None]:
        primary = self._targets[target_ref].strategies[0]
        if isinstance(primary, VisualTarget):
            return "canvas", None
        locator = primary.locator if isinstance(primary, WebTarget) else None
        if isinstance(locator, RoleLocator):
            return locator.role, locator.name
        if isinstance(locator, TableCellLocator) and locator.control is not None:
            return locator.control.role, locator.control.name
        if isinstance(locator, LabeledFieldLocator):
            return "textbox", locator.label
        return "cell", None
