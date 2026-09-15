from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

from playwright.async_api import ElementHandle, Frame, Page

from interface_ai_challenge.adapters.playwright import dom_scripts
from interface_ai_challenge.adapters.playwright.errors import surface_errors
from interface_ai_challenge.adapters.playwright.frames import FrameDirectory
from interface_ai_challenge.domain.elements import ElementDescriptor
from interface_ai_challenge.domain.geometry import Point, Rect, Screenshot, Size
from interface_ai_challenge.domain.targets import (
    FrameRef,
    LabeledFieldLocator,
    LabeledValueLocator,
    RoleLocator,
    TableCellLocator,
    TableColumnLocator,
    Target,
    VisualTarget,
    WebTarget,
)
from interface_ai_challenge.domain.values import ValueResolver
from interface_ai_challenge.ports.imaging import AssetStore, TemplateMatcher
from interface_ai_challenge.ports.surface import Candidate, ScreenCapture, Surface

SURFACE_FEATURES = frozenset({"web.frames", "web.tables", "visual.template"})
_MAX_FRAME_DEPTH = 8
_SETTLE_MS = 300
_SETTLE_TIMEOUT_MS = 10_000


class PlaywrightScreen:
    def __init__(self, page: Page, viewport: Size) -> None:
        self._page = page
        self._viewport = viewport
        self._revisions = itertools.count(1)

    def viewport(self) -> Size:
        return self._viewport

    @surface_errors
    async def capture(self) -> Screenshot:
        png = await self._page.screenshot(type="png", animations="disabled", caret="hide")
        return Screenshot(png=png, size=self._viewport, revision=next(self._revisions))


class PlaywrightRawInput:
    def __init__(self, page: Page) -> None:
        self._page = page

    @surface_errors
    async def click(self, point: Point, button: str) -> None:
        await self._page.mouse.click(point.x, point.y, button=button)  # type: ignore[arg-type]

    @surface_errors
    async def type_text(self, text: str) -> None:
        await self._page.keyboard.type(text)

    @surface_errors
    async def press_keys(self, keys: Sequence[str]) -> None:
        await self._page.keyboard.press("+".join(keys))

    @surface_errors
    async def scroll(self, point: Point, delta_x: int, delta_y: int) -> None:
        await self._page.mouse.move(point.x, point.y)
        await self._page.mouse.wheel(delta_x, delta_y)

    @surface_errors
    async def wait(self, milliseconds: int) -> None:
        await self._page.wait_for_timeout(milliseconds)

    @surface_errors
    async def settle(self) -> None:
        await self._page.wait_for_timeout(_SETTLE_MS)
        for frame in self._page.frames:
            await frame.wait_for_load_state("load", timeout=_SETTLE_TIMEOUT_MS)


class PlaywrightInspector:
    def __init__(self, page: Page, frames: FrameDirectory) -> None:
        self._page = page
        self._frames = frames

    @surface_errors
    async def describe_at(self, point: Point) -> ElementDescriptor | None:
        located = await self._element_at(point)
        return None if located is None else await self._describe(*located)

    @surface_errors
    async def describe_focused(self) -> ElementDescriptor | None:
        frame = self._page.main_frame
        for _ in range(_MAX_FRAME_DEPTH):
            element = (await frame.evaluate_handle(dom_scripts.ACTIVE_ELEMENT)).as_element()
            if element is None:
                return None
            if not await element.evaluate(dom_scripts.IS_IFRAME):
                return await self._describe(frame, element)
            child = await element.content_frame()
            if child is None:
                return None
            frame = child
        return None

    @surface_errors
    async def is_same_element(self, candidate: Candidate, point: Point) -> bool:
        if isinstance(candidate.handle, Point):
            return candidate.rect.contains(point)
        located = await self._element_at(point)
        if located is None:
            return False
        frame, element = located
        handle = candidate.handle
        if not isinstance(handle, tuple) or frame is not handle[0]:
            return False
        return bool(await frame.evaluate(dom_scripts.SAME_ELEMENT, [element, handle[1]]))

    async def _element_at(self, point: Point) -> tuple[Frame, ElementHandle] | None:
        frame = self._page.main_frame
        x, y = float(point.x), float(point.y)
        for _ in range(_MAX_FRAME_DEPTH):
            element = (await frame.evaluate_handle(dom_scripts.ELEMENT_FROM_POINT, [x, y])).as_element()
            if element is None:
                return None
            if not await element.evaluate(dom_scripts.IS_IFRAME):
                return frame, element
            child = await element.content_frame()
            box = await element.bounding_box()
            if child is None or box is None:
                return None
            border_x, border_y = await element.evaluate(dom_scripts.IFRAME_BORDER)
            x, y = point.x - box["x"] - border_x, point.y - box["y"] - border_y
            frame = child
        return None

    async def _describe(self, frame: Frame, element: ElementHandle) -> ElementDescriptor | None:
        box = await element.bounding_box()
        if box is None:
            return None
        data: dict[str, Any] = await element.evaluate(dom_scripts.DESCRIBE)
        frame_ref, named = self._frames.reference(frame)
        return ElementDescriptor(
            frame=frame_ref,
            frame_is_named=named,
            rect=Rect(x=box["x"], y=box["y"], width=box["width"], height=box["height"]),
            **data,
        )


class PlaywrightWebLocator:
    def __init__(self, frames: FrameDirectory) -> None:
        self._frames = frames

    @surface_errors
    async def locate(self, target: WebTarget, values: ValueResolver) -> tuple[Candidate, ...]:
        frame = self._frames.find(target.frame)
        if frame is None:
            return ()
        array = await frame.evaluate_handle(dom_scripts.RESOLVE, self._spec(target, values))
        candidates: list[Candidate] = []
        for handle in (await array.get_properties()).values():
            element = handle.as_element()
            if element is None:
                continue
            box = await element.bounding_box()
            if box is not None:
                rect = Rect(x=box["x"], y=box["y"], width=box["width"], height=box["height"])
                candidates.append(Candidate(rect=rect, handle=(frame, element)))
        return tuple(candidates)

    @staticmethod
    def _spec(target: WebTarget, values: ValueResolver) -> dict[str, Any]:
        locator = target.locator
        match locator:
            case RoleLocator(role=role, name=name):
                return {"strategy": "role", "role": role, "name": name}
            case LabeledFieldLocator(label=label) | LabeledValueLocator(label=label):
                return {"strategy": locator.strategy, "label": label}
            case TableColumnLocator(column=column):
                return {"strategy": "table_column", "column": column}
            case TableCellLocator():
                control = locator.control
                return {
                    "strategy": "table_cell",
                    "column": locator.column,
                    "row_key_column": locator.row_key_column,
                    "row_key": values.resolve(locator.row_key),
                    "control": {"role": control.role, "name": control.name} if control else None,
                }
        raise TypeError(f"unsupported locator {locator!r}")


class VisualLocator:
    def __init__(self, screen: ScreenCapture, matcher: TemplateMatcher, assets: AssetStore) -> None:
        self._screen = screen
        self._matcher = matcher
        self._assets = assets

    async def locate(self, target: VisualTarget) -> tuple[Candidate, ...]:
        if self._screen.viewport() != target.viewport:
            return ()
        screenshot = await self._screen.capture()
        template = self._assets.get(target.template_sha256)
        matches = self._matcher.find(template, screenshot.png, target.search_region, target.min_score)
        return tuple(
            Candidate(
                rect=match.rect,
                handle=Point(x=int(match.rect.x) + target.click_offset.x, y=int(match.rect.y) + target.click_offset.y),
            )
            for match in matches
        )


class CompositeTargetLocator:
    def __init__(self, web: PlaywrightWebLocator, visual: VisualLocator) -> None:
        self._web = web
        self._visual = visual

    async def locate(self, target: Target, values: ValueResolver) -> tuple[Candidate, ...]:
        if isinstance(target, VisualTarget):
            return await self._visual.locate(target)
        return await self._web.locate(target, values)


class PlaywrightElementActions:
    def __init__(self, page: Page) -> None:
        self._page = page

    @surface_errors
    async def click(self, candidate: Candidate, timeout_ms: int) -> None:
        if isinstance(candidate.handle, Point):
            await self._page.mouse.click(candidate.handle.x, candidate.handle.y)
            return
        await _element(candidate).click(timeout=timeout_ms)

    @surface_errors
    async def fill(self, candidate: Candidate, text: str, timeout_ms: int) -> None:
        await _element(candidate).fill(text, timeout=timeout_ms)

    @surface_errors
    async def press(self, candidate: Candidate, key: str, timeout_ms: int) -> None:
        await _element(candidate).press(key, timeout=timeout_ms)

    @surface_errors
    async def read_text(self, candidate: Candidate) -> str:
        return " ".join((await _element(candidate).inner_text()).split())

    @surface_errors
    async def read_field(self, candidate: Candidate) -> str:
        return await _element(candidate).input_value()


class PlaywrightProbe:
    def __init__(self, page: Page, frames: FrameDirectory) -> None:
        self._page = page
        self._frames = frames

    @surface_errors
    async def text_visible(self, text: str, frame: FrameRef | None) -> bool:
        if frame is not None:
            found = self._frames.find(frame)
            return found is not None and bool(await found.evaluate(dom_scripts.TEXT_VISIBLE, text))
        for candidate in self._frames.all_frames():
            if await candidate.evaluate(dom_scripts.TEXT_VISIBLE, text):
                return True
        return False

    @surface_errors
    async def dialog_titles(self) -> list[str]:
        titles: list[str] = []
        for frame in self._frames.all_frames():
            titles.extend(await frame.evaluate(dom_scripts.DIALOG_TITLES))
        return titles

    async def current_url(self) -> str:
        frames = self._frames.all_frames()
        return frames[-1].url if frames else self._page.url


class PlaywrightNavigator:
    def __init__(self, page: Page) -> None:
        self._page = page

    @surface_errors
    async def goto(self, url: str) -> None:
        await self._page.goto(url, wait_until="load")


def build_playwright_surface(page: Page, viewport: Size, matcher: TemplateMatcher, assets: AssetStore) -> Surface:
    frames = FrameDirectory(page)
    screen = PlaywrightScreen(page, viewport)
    return Surface(
        screen=screen,
        raw_input=PlaywrightRawInput(page),
        inspector=PlaywrightInspector(page, frames),
        locator=CompositeTargetLocator(PlaywrightWebLocator(frames), VisualLocator(screen, matcher, assets)),
        actions=PlaywrightElementActions(page),
        probe=PlaywrightProbe(page, frames),
        navigator=PlaywrightNavigator(page),
        features=SURFACE_FEATURES,
    )


def _element(candidate: Candidate) -> ElementHandle:
    handle = candidate.handle
    if not isinstance(handle, tuple):
        raise TypeError("visual targets only support clicks")
    element: ElementHandle = handle[1]
    return element
