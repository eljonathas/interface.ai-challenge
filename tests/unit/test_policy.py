import pytest
from support import configs

from interface_ai_challenge.application.policy_guard import ActionIntent, EffectClassifier, PolicyGuard
from interface_ai_challenge.domain.control import ActorKind
from interface_ai_challenge.domain.elements import ElementDescriptor
from interface_ai_challenge.domain.geometry import Rect
from interface_ai_challenge.domain.policy import UrlPolicy
from interface_ai_challenge.domain.steps import Effect
from interface_ai_challenge.domain.targets import FrameRef


@pytest.mark.parametrize(
    ("url", "allowed"),
    [
        ("http://127.0.0.1:8000/desk", True),
        ("http://127.0.0.1:8000/w/m/abc123/accounts?x=1", True),
        ("http://127.0.0.1:8001/desk", False),
        ("https://127.0.0.1:8000/desk", False),
        ("http://127.0.0.1:8000@evil.example/desk", False),
        ("http://evil.example/127.0.0.1:8000/desk", False),
        ("http://127.0.0.1:8000/admin", False),
        ("http://127.0.0.1:8000/w/../admin", False),
        ("http://127.0.0.1:8000/w/%2e%2e/admin", False),
        ("javascript:alert(1)", False),
        ("file:///etc/passwd", False),
        ("about:blank", True),
    ],
)
def test_url_policy_compares_parsed_components(url: str, allowed: bool) -> None:
    assert UrlPolicy(configs.policy()).allows(url) is allowed


def _intent(**overrides: object) -> ActionIntent:
    values: dict[str, object] = {"command": "click", "effect": Effect.NAVIGATION}
    values.update(overrides)
    return ActionIntent(**values)  # type: ignore[arg-type]


def test_guard_allows_allowlisted_action() -> None:
    assert PolicyGuard(configs.policy()).evaluate(_intent(), ActorKind.AUTOMATION).allowed


@pytest.mark.parametrize(
    "intent",
    [
        _intent(command="drag"),
        _intent(effect=Effect.IRREVERSIBLE),
        _intent(effect=Effect.UNKNOWN),
        _intent(command="keypress", effect=Effect.REVERSIBLE_INPUT, keys=("Control", "w")),
        _intent(command="navigate", url="http://evil.example/"),
        _intent(command="type", effect=Effect.REVERSIBLE_INPUT, text_length=10_000),
    ],
)
def test_guard_denies_outside_allowlist(intent: ActionIntent) -> None:
    decision = PolicyGuard(configs.policy()).evaluate(intent, ActorKind.AUTOMATION)
    assert not decision.allowed
    assert decision.reason


def test_humans_cannot_perform_irreversible_actions_either() -> None:
    decision = PolicyGuard(configs.policy()).evaluate(_intent(effect=Effect.IRREVERSIBLE), ActorKind.HUMAN)
    assert not decision.allowed


def _descriptor(role: str, name: str) -> ElementDescriptor:
    return ElementDescriptor(
        frame=FrameRef(names=("workspace",)),
        role=role,  # type: ignore[arg-type]
        name=name,
        rect=Rect(x=0, y=0, width=1, height=1),
    )


@pytest.mark.parametrize(
    ("role", "name", "effect"),
    [
        ("button", "Close account", Effect.IRREVERSIBLE),
        ("button", "Export file", Effect.IRREVERSIBLE),
        ("button", "Search", Effect.SUBMIT),
        ("link", "Open", Effect.NAVIGATION),
        ("textbox", "", Effect.REVERSIBLE_INPUT),
        ("canvas", "", Effect.NAVIGATION),
        ("cell", "$1.00", Effect.READ),
        ("other", "", Effect.UNKNOWN),
    ],
)
def test_effect_classification(role: str, name: str, effect: Effect) -> None:
    assert EffectClassifier(configs.policy()).for_click(_descriptor(role, name)) is effect


def test_enter_is_classified_as_submit_and_needs_focus() -> None:
    classifier = EffectClassifier(configs.policy())
    assert classifier.for_keys(("Enter",), _descriptor("textbox", "")) is Effect.SUBMIT
    assert classifier.for_keys(("Enter",), None) is Effect.UNKNOWN
    assert classifier.for_click(None) is Effect.UNKNOWN
