from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from interface_ai_challenge.domain.elements import ElementDescriptor
from interface_ai_challenge.domain.geometry import Point, Rect, Screenshot, Size
from interface_ai_challenge.domain.targets import FrameRef, Target
from interface_ai_challenge.domain.values import ValueResolver


@dataclass(frozen=True)
class Candidate:
    """A resolved control on the live surface. `handle` is adapter-private and short-lived."""

    rect: Rect
    handle: object


class ScreenCapture(Protocol):
    async def capture(self) -> Screenshot: ...

    def viewport(self) -> Size: ...


class RawInput(Protocol):
    async def click(self, point: Point, button: str) -> None: ...

    async def type_text(self, text: str) -> None: ...

    async def press_keys(self, keys: Sequence[str]) -> None: ...

    async def scroll(self, point: Point, delta_x: int, delta_y: int) -> None: ...

    async def wait(self, milliseconds: int) -> None: ...

    async def settle(self) -> None: ...


class ElementInspector(Protocol):
    async def describe_at(self, point: Point) -> ElementDescriptor | None: ...

    async def describe_focused(self) -> ElementDescriptor | None: ...

    async def is_same_element(self, candidate: Candidate, point: Point) -> bool: ...


class TargetLocator(Protocol):
    async def locate(self, target: Target, values: ValueResolver) -> tuple[Candidate, ...]: ...


class ElementActions(Protocol):
    async def click(self, candidate: Candidate, timeout_ms: int) -> None: ...

    async def fill(self, candidate: Candidate, text: str, timeout_ms: int) -> None: ...

    async def press(self, candidate: Candidate, key: str, timeout_ms: int) -> None: ...

    async def read_text(self, candidate: Candidate) -> str: ...

    async def read_field(self, candidate: Candidate) -> str: ...


class PageProbe(Protocol):
    async def text_visible(self, text: str, frame: FrameRef | None) -> bool: ...

    async def dialog_titles(self) -> list[str]: ...

    async def current_url(self) -> str: ...


class Navigator(Protocol):
    async def goto(self, url: str) -> None: ...


@dataclass(frozen=True)
class Surface:
    screen: ScreenCapture
    raw_input: RawInput
    inspector: ElementInspector
    locator: TargetLocator
    actions: ElementActions
    probe: PageProbe
    navigator: Navigator
    features: frozenset[str]
