"""End-to-end checks against the real demo application in Chromium.

The discovery here is driven by a scripted oracle that plays the model port so the pipeline can be tested
without API calls. It is not, and is never presented as, evidence of a real LLM-driven discovery.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest
from support import configs

from interface_ai_challenge.adapters.file_store import CapabilityFileRepository, FileAssetStore
from interface_ai_challenge.application.discovery.runner import DiscoveryBudget
from interface_ai_challenge.bootstrap import SessionServices, open_session, run_discovery, run_replay
from interface_ai_challenge.domain.computer import ClickCommand, ComputerCommand, KeypressCommand, TypeCommand
from interface_ai_challenge.domain.geometry import Point, Screenshot
from interface_ai_challenge.domain.results import RunResult
from interface_ai_challenge.domain.steps import FillAction
from interface_ai_challenge.domain.targets import (
    FrameRef,
    LabeledFieldLocator,
    LabeledValueLocator,
    RoleLocator,
    TableCellLocator,
    VisualTarget,
    WebTarget,
)
from interface_ai_challenge.domain.values import InputRef, LiteralValue, ValueResolver
from interface_ai_challenge.ports.model import (
    AgentFeedback,
    AgentTask,
    AgentTurn,
    FinishSignal,
    ReportIdentitySignal,
    ReportOutputSignal,
)
from interface_ai_challenge.settings import RuntimeSettings

pytestmark = pytest.mark.browser

WORKSPACE = FrameRef(names=("workspace",))
NO_VALUES = ValueResolver({}, {})


async def _center(session: SessionServices, locator: object, values: ValueResolver = NO_VALUES) -> Point:
    candidates = await session.surface.locator.locate(WebTarget(frame=WORKSPACE, locator=locator), values)  # type: ignore[arg-type]
    assert len(candidates) == 1, f"oracle expected one element for {locator!r}"
    return candidates[0].rect.center()


async def _canvas_accounts_button(session: SessionServices) -> Point:
    branch = (
        await session.surface.locator.locate(
            WebTarget(frame=WORKSPACE, locator=LabeledValueLocator(label="Branch")), NO_VALUES
        )
    )[0].rect
    for y in range(int(branch.y + branch.height), int(branch.y + branch.height) + 200, 4):
        for x in range(0, 900, 20):
            descriptor = await session.surface.inspector.describe_at(Point(x=x, y=y))
            if descriptor is not None and descriptor.is_canvas:
                return Point(x=int(descriptor.rect.x) + 180, y=int(descriptor.rect.y) + 22)
    raise AssertionError("canvas toolbar not found")


class OracleAgent:
    """Scripted test double for the model port (not an LLM)."""

    provider = "scripted-oracle"
    model = "none"

    def __init__(self, member_id: str) -> None:
        self._values = ValueResolver({"member_id": member_id}, {})
        self._session: SessionServices | None = None
        self._turn = 0
        self.feedback: list[AgentFeedback] = []

    def attach(self, session: SessionServices) -> None:
        self._session = session

    async def begin(self, task: AgentTask, screenshot: Screenshot) -> AgentTurn:
        return await self._next()

    async def respond(self, feedback: AgentFeedback) -> AgentTurn:
        self.feedback.append(feedback)
        return await self._next()

    async def _next(self) -> AgentTurn:
        session = self._session
        assert session is not None
        self._turn += 1
        savings = LiteralValue(value="Savings")
        match self._turn:
            case 1:
                return _commands(ClickCommand(point=await _center(session, LabeledFieldLocator(label="Member ID"))))
            case 2:
                return _commands(TypeCommand(text="{{inputs.member_id}}"))
            case 3:
                return _commands(ClickCommand(point=await _center(session, RoleLocator(role="button", name="Search"))))
            case 4:
                open_link = TableCellLocator(
                    column="Action",
                    row_key_column="Member ID",
                    row_key=InputRef(name="member_id"),
                    control=RoleLocator(role="link", name="Open"),
                )
                return _commands(ClickCommand(point=await _center(session, open_link, self._values)))
            case 5:
                return _commands(ClickCommand(point=await _canvas_accounts_button(session)))
            case 6:
                columns = {"account_type": "Account type", "balance": "Current balance", "currency": "Currency"}
                signals = []
                for output, column in columns.items():
                    point = await _center(
                        session, TableCellLocator(column=column, row_key_column="Account type", row_key=savings)
                    )
                    signals.append(ReportOutputSignal(call_id=f"report-{output}", output=output, point=point))
                identity = await _center(session, LabeledValueLocator(label="Member ID"))
                signals.append(ReportIdentitySignal(call_id="identity", input="member_id", point=identity))
                return AgentTurn(commands=(), signals=tuple(signals))
            case _:
                return AgentTurn(commands=(), signals=(FinishSignal(call_id="finish", summary="done"),))


def _commands(*commands: ComputerCommand) -> AgentTurn:
    return AgentTurn(commands=commands, signals=())


@pytest.fixture(scope="module")
def capability(demo_app_session: Callable[[str], str], tmp_path_factory: pytest.TempPathFactory) -> Path:
    base_url = demo_app_session("normal")
    output = tmp_path_factory.mktemp("capability")
    agent = OracleAgent("M-10023")
    report = asyncio.run(
        run_discovery(
            agent=agent,
            contract=configs.contract(),
            inputs={"member_id": "M-10023"},
            entry_url=f"{base_url}/desk",
            output_dir=output,
            policy=configs.policy(base_url),
            profile=configs.profile(),
            runtime=RuntimeSettings(),
            budget=DiscoveryBudget(max_turns=12),
            session_hook=agent.attach,
        )
    )
    assert report.completed, (report.reason, report.problems)
    assert report.artifact_path is not None
    return report.artifact_path


def _replay(
    capability: Path,
    base_url: str,
    member_id: str,
    run_dir: Path,
    enable_operator: bool = False,
    hook: Callable[[SessionServices], None] | None = None,
) -> RunResult:
    return asyncio.run(
        run_replay(
            artifact_path=capability,
            inputs={"member_id": member_id},
            bindings={"base_url": base_url},
            run_dir=run_dir,
            policy=configs.policy(base_url),
            redaction=configs.profile().redaction,
            runtime=RuntimeSettings(enable_operator=enable_operator, intervention_timeout_seconds=30),
            session_hook=hook,
        )
    )


def test_discovery_compiles_a_parameterized_capability(capability: Path) -> None:
    artifact = CapabilityFileRepository().load(capability)
    assert [step.action.kind for step in artifact.steps] == ["navigate", "fill", "click", "click", "click"]
    assert any(isinstance(s, VisualTarget) for target in artifact.targets.values() for s in target.strategies)
    assert "M-10023" not in capability.read_text()
    assert "1,250.45" not in capability.read_text()


def test_replay_with_other_member_returns_its_balance(
    capability: Path, demo_app_session: Callable[[str], str], tmp_path: Path
) -> None:
    result = _replay(capability, demo_app_session("normal"), "M-20417", tmp_path)
    assert result.status == "success", result
    assert result.outputs == {"account_type": "savings", "balance": "12980.07", "currency": "USD"}
    persisted = (tmp_path / "events.jsonl").read_text() + (tmp_path / "result.json").read_text()
    assert "M-20417" not in persisted and "12980.07" not in persisted


@pytest.mark.parametrize(
    ("member_id", "code"),
    [("M-99999", "member_not_found"), ("M-30555", "account_not_found"), ("10023", "validation_rejected")],
)
def test_business_outcomes(
    capability: Path, demo_app_session: Callable[[str], str], tmp_path: Path, member_id: str, code: str
) -> None:
    result = _replay(capability, demo_app_session("normal"), member_id, tmp_path)
    assert result.status == "business_outcome" and result.code == code, result


@pytest.mark.parametrize("scenario", ["maintenance", "transient_error", "slow", "canvas_shift"])
def test_recoverable_conditions_and_visual_relocation(
    capability: Path, demo_app: Callable[[str], str], tmp_path: Path, scenario: str
) -> None:
    result = _replay(capability, demo_app(scenario), "M-10023", tmp_path)
    assert result.status == "success", result
    assert result.outputs["balance"] == "1250.45"


@pytest.mark.parametrize(
    ("scenario", "code"),
    [
        ("permission_denied", "permission_denied"),
        ("duplicate_canvas_control", "target_ambiguous"),
        ("unknown_dialog", "escalation_unavailable"),
    ],
)
def test_hard_failures_are_explicit(
    capability: Path, demo_app: Callable[[str], str], tmp_path: Path, scenario: str, code: str
) -> None:
    result = _replay(capability, demo_app(scenario), "M-10023", tmp_path)
    assert result.status == "failure" and result.code == code, result
    assert result.evidence_ref and (tmp_path / result.evidence_ref).exists()


def test_operator_takes_over_live_session_and_hands_back(
    capability: Path, demo_app: Callable[[str], str], tmp_path: Path
) -> None:
    tasks: list[asyncio.Task[None]] = []

    async def operate(session: SessionServices) -> None:
        operator = session.operator
        assert operator is not None
        while operator.status()["intervention"] is None:
            await asyncio.sleep(0.05)
        token = await operator.claim("operator-test")
        for label in ("Username", "Password"):
            point = await _center(session, LabeledFieldLocator(label=label))
            await operator.act(
                "operator-test", token.epoch, (await operator.screenshot()).revision, ClickCommand(point=point)
            )
            await operator.act("operator-test", token.epoch, -1, TypeCommand(text="demo"))
        sign_in = await _center(session, RoleLocator(role="button", name="Sign in"))
        await operator.act(
            "operator-test", token.epoch, (await operator.screenshot()).revision, ClickCommand(point=sign_in)
        )
        await session.surface.raw_input.settle()
        await operator.hand_back("operator-test")

    def hook(session: SessionServices) -> None:
        tasks.append(asyncio.get_running_loop().create_task(operate(session)))

    result = _replay(capability, demo_app("session_expired"), "M-20417", tmp_path, enable_operator=True, hook=hook)
    assert result.status == "success" and result.assisted, result
    events = [json.loads(line)["event"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events.count("human_action") == 5
    assert {"intervention_requested", "control_claimed", "control_returned", "resume_reconciled"} <= set(events)
    assert "demo" not in (tmp_path / "events.jsonl").read_text()


async def test_prompt_injection_link_cannot_leave_the_allowlist(demo_app: Callable[[str], str], tmp_path: Path) -> None:
    base_url = demo_app("injection")
    async with open_session(
        "discovery",
        tmp_path,
        configs.policy(base_url),
        configs.profile().redaction,
        RuntimeSettings(),
        FileAssetStore(tmp_path),
    ) as session:
        await session.commands.navigate(session.control.automation_token(), f"{base_url}/desk")
        link = await _center(session, RoleLocator(role="link", name="the external help desk"))
        await session.commands.execute(session.control.automation_token(), ClickCommand(point=link))
        await session.surface.raw_input.settle()
    blocked = [
        json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines() if '"request_blocked"' in line
    ]
    assert any("example.com" in event["data"]["url"] for event in blocked)


@pytest.mark.parametrize(
    "typing",
    [
        (TypeCommand(text="M-"), TypeCommand(text="10023")),
        (
            TypeCommand(text="M-"),
            TypeCommand(text="10024"),
            KeypressCommand(keys=("Backspace",)),
            TypeCommand(text="3"),
        ),
    ],
)
async def test_edited_input_compiles_and_replays_for_another_member(
    demo_app: Callable[[str], str], tmp_path: Path, typing: tuple[ComputerCommand, ...]
) -> None:
    class EditingOracle(OracleAgent):
        async def _next(self) -> AgentTurn:
            turn = await super()._next()
            return _commands(*typing) if any(isinstance(c, TypeCommand) for c in turn.commands) else turn

    base_url = demo_app("normal")
    agent = EditingOracle("M-10023")
    report = await run_discovery(
        agent,
        configs.contract(),
        {"member_id": "M-10023"},
        f"{base_url}/desk",
        tmp_path / "capability",
        configs.policy(base_url),
        configs.profile(),
        RuntimeSettings(),
        session_hook=agent.attach,
    )
    assert report.completed, (report.reason, report.problems)
    assert report.artifact_path is not None
    artifact = CapabilityFileRepository().load(report.artifact_path)
    fills = [step.action for step in artifact.steps if isinstance(step.action, FillAction)]
    assert len(fills) == 1 and fills[0].value == InputRef(name="member_id")
    assert "M-10023" not in report.artifact_path.read_text()
    result = await run_replay(
        report.artifact_path,
        {"member_id": "M-20417"},
        {"base_url": base_url},
        tmp_path / "replay",
        configs.policy(base_url),
        configs.profile().redaction,
        RuntimeSettings(),
    )
    assert result.status == "success" and result.outputs["balance"] == "12980.07", result


async def test_search_evidence_masks_names_and_ids_in_every_result_row(
    demo_app: Callable[[str], str], tmp_path: Path
) -> None:
    base_url = demo_app("normal")
    async with open_session(
        "discovery",
        tmp_path,
        configs.policy(base_url),
        configs.profile().redaction,
        RuntimeSettings(),
        FileAssetStore(tmp_path / "assets"),
    ) as session:
        token = session.control.automation_token()
        await session.commands.navigate(token, f"{base_url}/desk")
        await session.commands.execute(
            token, ClickCommand(point=await _center(session, LabeledFieldLocator(label="Member ID")))
        )
        await session.commands.execute(token, TypeCommand(text="M-10023"))
        await session.commands.execute(
            token, ClickCommand(point=await _center(session, RoleLocator(role="button", name="Search")))
        )
        await session.surface.raw_input.settle()
        screenshot = await session.evidence_screen.capture()
        saved = session.evidence.attach_image("search-results", screenshot.png)
        masked = cv2.imdecode(np.frombuffer((tmp_path / saved).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        # Inspect each known result independently of the table-column mask strategy.
        for member in ("M-10023", "M-100231"):
            for column in ("Member ID", "Name", "Branch"):
                target = WebTarget(
                    frame=WORKSPACE,
                    locator=TableCellLocator(
                        column=column,
                        row_key_column="Member ID",
                        row_key=LiteralValue(value=member),
                    ),
                )
                candidates = await session.surface.locator.locate(target, NO_VALUES)
                assert len(candidates) == 1
                rect = candidates[0].rect
                x, y, width, height = int(rect.x), int(rect.y), int(rect.width), int(rect.height)
                pixels = masked[y + 4 : y + height - 4, x + 4 : x + width - 4]
                assert pixels.size > 0
                if column == "Branch":
                    assert np.any(pixels != 0)
                else:
                    assert np.all(pixels == 0), f"unmasked {column} for {member}"


async def test_discovery_rejects_a_balance_from_another_account_and_can_correct_it(
    demo_app: Callable[[str], str], tmp_path: Path
) -> None:
    class CrossRowOracle(OracleAgent):
        async def _next(self) -> AgentTurn:
            turn = await super()._next()
            if self._turn == 6:
                assert self._session is not None
                point = await _center(
                    self._session,
                    TableCellLocator(
                        column="Current balance", row_key_column="Account type", row_key=LiteralValue(value="Checking")
                    ),
                )
                return AgentTurn(
                    commands=(),
                    signals=tuple(
                        ReportOutputSignal(call_id=s.call_id, output="balance", point=point)
                        if isinstance(s, ReportOutputSignal) and s.output == "balance"
                        else s
                        for s in turn.signals
                    ),
                )
            if self._turn == 7:
                assert self._session is not None
                assert "same table row" in self.feedback[-1].signal_replies["report-balance"]
                point = await _center(
                    self._session,
                    TableCellLocator(
                        column="Current balance", row_key_column="Account type", row_key=LiteralValue(value="Savings")
                    ),
                )
                return AgentTurn(
                    commands=(), signals=(ReportOutputSignal(call_id="correct-balance", output="balance", point=point),)
                )
            return turn

    base_url = demo_app("normal")
    agent = CrossRowOracle("M-10023")
    report = await run_discovery(
        agent,
        configs.contract(),
        {"member_id": "M-10023"},
        f"{base_url}/desk",
        tmp_path / "capability",
        configs.policy(base_url),
        configs.profile(),
        RuntimeSettings(),
        session_hook=agent.attach,
    )
    assert report.completed and report.artifact_path is not None, report
    result = await run_replay(
        report.artifact_path,
        {"member_id": "M-20417"},
        {"base_url": base_url},
        tmp_path / "replay",
        configs.policy(base_url),
        configs.profile().redaction,
        RuntimeSettings(),
    )
    assert result.status == "success" and result.outputs["balance"] == "12980.07", result
