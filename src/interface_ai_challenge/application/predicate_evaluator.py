from __future__ import annotations

from interface_ai_challenge.application.target_resolution import NamedTargetResolver
from interface_ai_challenge.domain.errors import SurfaceError
from interface_ai_challenge.domain.predicates import (
    Absent,
    AllOf,
    AnyOf,
    DialogVisible,
    FieldEquals,
    Predicate,
    TextEquals,
    TextVisible,
    Visible,
)
from interface_ai_challenge.domain.values import ValueResolver
from interface_ai_challenge.ports.surface import ElementActions, PageProbe


def normalize_text(text: str) -> str:
    return " ".join(text.split())


class PredicateEvaluator:
    def __init__(
        self,
        resolver: NamedTargetResolver,
        actions: ElementActions,
        probe: PageProbe,
        values: ValueResolver,
    ) -> None:
        self._resolver = resolver
        self._actions = actions
        self._probe = probe
        self._values = values

    async def holds(self, predicate: Predicate) -> bool:
        try:
            return await self._evaluate(predicate)
        except SurfaceError:
            return False

    async def candidate_count(self, target_ref: str) -> int:
        try:
            return len((await self._resolver.resolve(target_ref, self._values)).candidates)
        except SurfaceError:
            return 0

    async def _evaluate(self, predicate: Predicate) -> bool:
        match predicate:
            case Visible(target_ref=ref):
                return await self.candidate_count(ref) == 1
            case Absent(target_ref=ref):
                return await self.candidate_count(ref) == 0
            case TextEquals(target_ref=ref, value=value):
                candidate = (await self._resolver.resolve(ref, self._values)).unique
                if candidate is None:
                    return False
                actual = normalize_text(await self._actions.read_text(candidate))
                return actual == normalize_text(self._values.resolve(value))
            case FieldEquals(target_ref=ref, value=value):
                candidate = (await self._resolver.resolve(ref, self._values)).unique
                if candidate is None:
                    return False
                return await self._actions.read_field(candidate) == self._values.resolve(value)
            case TextVisible(text=text, frame=frame):
                return await self._probe.text_visible(text, frame)
            case DialogVisible():
                return bool(await self._probe.dialog_titles())
            case AllOf(predicates=children):
                for child in children:
                    if not await self._evaluate(child):
                        return False
                return True
            case AnyOf(predicates=children):
                for child in children:
                    if await self._evaluate(child):
                        return True
                return False
        raise TypeError(f"unsupported predicate {predicate!r}")
