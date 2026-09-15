import io
from unittest.mock import AsyncMock, Mock

import pytest
from support.evidence import InMemoryEvidenceSink
from support.fake_surface import FakeScreen

from interface_ai_challenge.adapters.console_log import ConsoleEvidenceSink, print_model_turn
from interface_ai_challenge.application.control import SessionControl
from interface_ai_challenge.application.handoff import EscalationContext, InterventionCoordinator
from interface_ai_challenge.application.redaction import MaskedScreenCapture, RedactingEvidenceSink, Redactor
from interface_ai_challenge.bootstrap import print_intervention
from interface_ai_challenge.domain.errors import SurfaceError
from interface_ai_challenge.domain.geometry import Rect
from interface_ai_challenge.domain.intervention import InterventionDecision
from interface_ai_challenge.domain.targets import LabeledValueLocator, WebTarget
from interface_ai_challenge.domain.values import ValueResolver
from interface_ai_challenge.ports.model import AgentTurn
from interface_ai_challenge.ports.surface import Candidate


def test_redactor_replaces_registered_values_recursively() -> None:
    redactor = Redactor()
    redactor.register("input.member_id", "M-10023")
    redactor.register("output.balance", "1250.45")
    redacted = redactor.redact({"message": "member M-10023 has 1250.45", "items": [("M-10023",)], "count": 3})
    assert redacted == {
        "message": "member [REDACTED:input.member_id] has [REDACTED:output.balance]",
        "items": [["[REDACTED:input.member_id]"]],
        "count": 3,
    }


def test_redacting_sink_applies_before_persistence() -> None:
    inner = InMemoryEvidenceSink()
    redactor = Redactor()
    sink = RedactingEvidenceSink(inner, redactor)
    redactor.register("input.member_id", "M-20417")
    sink.emit("event", {"note": "typed M-20417"})
    sink.write_document("result", {"outputs": {"member": "M-20417"}})
    assert "M-20417" not in inner.serialized()


def test_very_short_values_are_not_used_as_patterns() -> None:
    redactor = Redactor()
    redactor.register("input.flag", "1")
    assert redactor.redact("step 1 of 10") == "step 1 of 10"


@pytest.mark.parametrize("partial_match", [False, True])
async def test_masking_errors_never_return_an_unmasked_or_partially_masked_image(partial_match: bool) -> None:
    regions = [WebTarget(locator=LabeledValueLocator(label=label)) for label in ("Name", "SSN")]
    candidate = Candidate(rect=Rect(x=10, y=10, width=20, height=10), handle=None)
    locator = Mock(
        locate=AsyncMock(
            side_effect=[(candidate,) if partial_match else (), SurfaceError("frame detached: sensitive detail")]
        )
    )
    editor = Mock()
    capture = MaskedScreenCapture(FakeScreen(), locator, editor, regions, ValueResolver({}, {}))
    with pytest.raises(SurfaceError, match="^screenshot redaction could not be completed$"):
        await capture.capture()
    editor.mask.assert_not_called()


async def test_absent_sensitive_regions_are_not_confused_with_locator_errors() -> None:
    screen = FakeScreen()
    region = WebTarget(locator=LabeledValueLocator(label="SSN"))
    capture = MaskedScreenCapture(
        screen, Mock(locate=AsyncMock(return_value=())), Mock(), [region], ValueResolver({}, {})
    )
    assert (await capture.capture()).png == (await screen.capture()).png


def test_console_shows_redacted_events_and_model_text_without_persisting_it() -> None:
    persisted, stream = InMemoryEvidenceSink(), io.StringIO()
    redactor = Redactor()
    redactor.register("input.member_id", "M-10023")
    RedactingEvidenceSink(ConsoleEvidenceSink(persisted, "discovery", stream), redactor).emit(
        "agent_action_executed", {"typed": "M-10023"}
    )
    print_model_turn(
        AgentTurn(commands=(), signals=(), message="Alice Carter", reasoning=("Open the savings row.",)), stream
    )
    shown = stream.getvalue()
    assert "[REDACTED:input.member_id]" in shown and "M-10023" not in shown
    assert "Open the savings row." in shown and "Alice Carter" in shown
    assert "Alice Carter" not in persisted.serialized() and "M-10023" not in persisted.serialized()


async def test_private_handoff_reason_is_live_only(capsys: pytest.CaptureFixture[str]) -> None:
    evidence = InMemoryEvidenceSink()
    channel = Mock(publish=AsyncMock(), wait_for_decision=AsyncMock(return_value=InterventionDecision.ABORT))
    coordinator = InterventionCoordinator("discovery-test", SessionControl(), channel, FakeScreen(), evidence, 1)
    await coordinator.escalate(
        EscalationContext(
            mode="discovery",
            subject="member.savings_balance",
            step_id=None,
            reason="Cannot read Alice Carter's balance",
            resume_behavior="observe again",
            evidence_reason="model_requested_help",
        )
    )
    request = channel.publish.call_args.args[0]
    assert "Alice Carter" in request.reason  # Available to the operator in memory.
    assert "Alice Carter" not in evidence.serialized()
    print_intervention(request, None)
    assert "Alice Carter" not in capsys.readouterr().err
