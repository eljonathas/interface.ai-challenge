"""An in-memory stand-in for the demo application, reachable only through the surface ports.

It lets the replay engine, handlers and handoff be tested without a browser. Keys are derived from
target specifications exactly like a real locator would interpret them, so wrong targets do not resolve.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

from interface_ai_challenge.domain.elements import ElementDescriptor
from interface_ai_challenge.domain.geometry import Point, Rect, Screenshot, Size
from interface_ai_challenge.domain.targets import (
    FrameRef,
    LabeledFieldLocator,
    LabeledValueLocator,
    RoleLocator,
    TableCellLocator,
    Target,
    VisualTarget,
)
from interface_ai_challenge.domain.values import ValueResolver
from interface_ai_challenge.ports.surface import Candidate, Surface

VIEWPORT = Size(width=1280, height=800)
ACCOUNTS = {
    "M-10023": {"Checking": "$3,210.00", "Savings": "$1,250.45"},
    "M-100231": {"Savings": "$88.10"},
    "M-20417": {"Checking": "$540.00", "Savings": "$12,980.07"},
    "M-30555": {"Checking": "$75.20"},
}


def target_key(target: Target, values: ValueResolver) -> str:
    if isinstance(target, VisualTarget):
        return "visual:accounts"
    locator = target.locator
    if isinstance(locator, RoleLocator):
        return f"{locator.role}:{locator.name}"
    if isinstance(locator, LabeledFieldLocator):
        return f"field:{locator.label}"
    if isinstance(locator, LabeledValueLocator):
        return f"value:{locator.label}"
    assert isinstance(locator, TableCellLocator)
    key = f"cell:{locator.column}:{locator.row_key_column}={values.resolve(locator.row_key)}"
    if locator.control is not None:
        key += f":{locator.control.role}:{locator.control.name}"
    return key


class FakeApp:
    def __init__(self, scenario: str = "normal") -> None:
        self.scenario = scenario
        self.page = "blank"
        self.field = ""
        self.query = ""
        self.member: str | None = None
        self.notice = False
        self.dialog = False
        self.clicks: list[str] = []
        self._consumed: set[str] = set()

    def first_time(self, flag: str) -> bool:
        if flag in self._consumed:
            return False
        self._consumed.add(flag)
        return True

    def navigate(self) -> None:
        self.page, self.field = "search", ""
        self.notice = self.scenario == "maintenance" and self.first_time("notice")

    def elements(self) -> dict[str, list[str]]:
        if self.page == "search":
            elements = {"field:Member ID": [self.field], "button:Search": ["Search"]}
            if self.notice:
                elements["button:Dismiss"] = ["Dismiss"]
            if self.scenario == "irreversible":
                elements["button:Close account"] = ["Close account"]
            return elements
        if self.page == "results":
            return {
                f"cell:Action:Member ID={member}:link:Open": ["Open"]
                for member in ACCOUNTS
                if self.query and member.startswith(self.query)
            }
        if self.page == "member":
            copies = 2 if self.scenario == "duplicate_visual" else 1
            return {"value:Member ID": [self.member or ""], "visual:accounts": ["Accounts"] * copies}
        if self.page == "accounts":
            balances = ACCOUNTS[self.member or ""]
            elements = {"value:Member ID": [self.member or ""]}
            if "Savings" in balances:
                elements["cell:Account type:Account type=Savings"] = ["Savings"]
                elements["cell:Current balance:Account type=Savings"] = [balances["Savings"]]
                elements["cell:Currency:Account type=Savings"] = ["USD"]
            return elements
        if self.page == "transient":
            return {"link:Retry": ["Retry"]}
        return {}

    def text(self) -> str:
        if self.page == "results":
            if not self.query.startswith("M-"):
                return "Search results Invalid member ID format."
            if not self.elements() or self.scenario == "conflict":
                return "Search results No members match your search."
            return "Search results"
        return {
            "accounts": "Accounts Account type Current balance",
            "expired": "Your session has expired. Please sign in again.",
            "denied": "Access denied: your role cannot view account balances.",
            "transient": "Temporary read error while loading results.",
        }.get(self.page, "")

    def dialogs(self) -> list[str]:
        if self.page == "search" and self.notice:
            return ["Maintenance notice"]
        if self.page == "member" and self.dialog:
            return ["Export confirmation"]
        return []

    def click(self, key: str) -> None:
        self.clicks.append(key)
        if self.dialogs() and key != "button:Dismiss":
            raise_blocked()
        if key == "button:Search":
            self._submit_search()
        elif key == "button:Dismiss":
            self.notice = False
        elif key == "link:Retry":
            self.page = "results"
        elif key.startswith("cell:Action:Member ID="):
            self.member = key.split("=", 1)[1].split(":", 1)[0]
            self.page = "member"
            self.dialog = self.scenario == "unknown_dialog" and self.first_time("dialog")
        elif key == "visual:accounts":
            self.page = "denied" if self.scenario == "permission_denied" else "accounts"

    def raw_click(self) -> None:
        if self.page == "expired":
            self.page, self.field = "search", ""
        elif self.page == "member" and self.dialog:
            self.dialog = False

    def _submit_search(self) -> None:
        self.query = self.field
        if self.scenario == "session_expired" and self.first_time("expired"):
            self.page = "expired"
        elif self.scenario == "transient_error" and self.first_time("transient"):
            self.page = "transient"
        else:
            self.page = "results"


def raise_blocked() -> None:
    from interface_ai_challenge.domain.errors import SurfaceError

    raise SurfaceError("element is covered by a dialog")


class FakeScreen:
    def __init__(self) -> None:
        self._revisions = itertools.count(1)

    def viewport(self) -> Size:
        return VIEWPORT

    async def capture(self) -> Screenshot:
        return Screenshot(png=b"\x89PNG-fake", size=VIEWPORT, revision=next(self._revisions))


class FakeRawInput:
    def __init__(self, app: FakeApp) -> None:
        self._app = app

    async def click(self, point: Point, button: str) -> None:
        self._app.raw_click()

    async def type_text(self, text: str) -> None:
        return None

    async def press_keys(self, keys: Sequence[str]) -> None:
        return None

    async def scroll(self, point: Point, delta_x: int, delta_y: int) -> None:
        return None

    async def wait(self, milliseconds: int) -> None:
        return None

    async def settle(self) -> None:
        return None


class FakeInspector:
    async def describe_at(self, point: Point) -> ElementDescriptor | None:
        return ElementDescriptor(
            frame=FrameRef(names=("workspace",)),
            role="button",
            name="Sign in",
            rect=Rect(x=point.x - 5, y=point.y - 5, width=10, height=10),
        )

    async def describe_focused(self) -> ElementDescriptor | None:
        return None

    async def is_same_element(self, candidate: Candidate, point: Point) -> bool:
        return False


class FakeLocator:
    def __init__(self, app: FakeApp) -> None:
        self._app = app

    async def locate(self, target: Target, values: ValueResolver) -> tuple[Candidate, ...]:
        key = target_key(target, values)
        count = len(self._app.elements().get(key, []))
        return tuple(Candidate(rect=Rect(x=10, y=10 + i, width=20, height=10), handle=(key, i)) for i in range(count))


class FakeActions:
    def __init__(self, app: FakeApp) -> None:
        self._app = app

    async def click(self, candidate: Candidate, timeout_ms: int) -> None:
        self._app.click(_key(candidate))

    async def fill(self, candidate: Candidate, text: str, timeout_ms: int) -> None:
        self._app.field = text

    async def press(self, candidate: Candidate, key: str, timeout_ms: int) -> None:
        if key == "Enter":
            self._app.click("button:Search")

    async def read_text(self, candidate: Candidate) -> str:
        key, index = candidate.handle  # type: ignore[misc]
        return self._app.elements()[key][index]

    async def read_field(self, candidate: Candidate) -> str:
        return self._app.field


class FakeProbe:
    def __init__(self, app: FakeApp) -> None:
        self._app = app

    async def text_visible(self, text: str, frame: FrameRef | None) -> bool:
        return text in self._app.text()

    async def dialog_titles(self) -> list[str]:
        return self._app.dialogs()

    async def current_url(self) -> str:
        return f"http://127.0.0.1:8000/fake/{self._app.page}"


class FakeNavigator:
    def __init__(self, app: FakeApp) -> None:
        self._app = app

    async def goto(self, url: str) -> None:
        self._app.navigate()


def fake_surface(app: FakeApp) -> Surface:
    return Surface(
        screen=FakeScreen(),
        raw_input=FakeRawInput(app),
        inspector=FakeInspector(),
        locator=FakeLocator(app),
        actions=FakeActions(app),
        probe=FakeProbe(app),
        navigator=FakeNavigator(app),
        features=frozenset({"web.frames", "web.tables", "visual.template"}),
    )


def _key(candidate: Candidate) -> str:
    key, _ = candidate.handle  # type: ignore[misc]
    return str(key)
