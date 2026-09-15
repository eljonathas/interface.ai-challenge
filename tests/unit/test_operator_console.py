import re

from fastapi.testclient import TestClient
from support import configs
from support.evidence import InMemoryEvidenceSink
from support.fake_surface import FakeApp, fake_surface

from interface_ai_challenge.adapters.operator_console.app import create_operator_app
from interface_ai_challenge.application.commands import ComputerCommandExecutor
from interface_ai_challenge.application.control import SessionControl
from interface_ai_challenge.application.guarded_executor import GuardedExecutor
from interface_ai_challenge.application.intervention_inbox import InMemoryInterventionInbox
from interface_ai_challenge.application.operator_service import OperatorService
from interface_ai_challenge.application.policy_guard import EffectClassifier, PolicyGuard


def _client() -> TestClient:
    surface = fake_surface(FakeApp())
    policy = configs.policy()
    evidence = InMemoryEvidenceSink()
    control = SessionControl()
    executor = GuardedExecutor(control, PolicyGuard(policy), evidence)
    service = OperatorService(
        control,
        InMemoryInterventionInbox(),
        ComputerCommandExecutor(surface, executor, EffectClassifier(policy)),
        surface.screen,
        evidence,
    )
    return TestClient(create_operator_app(service, "local-token"), base_url="http://127.0.0.1:8765")


def test_console_requires_the_printed_token() -> None:
    client = _client()
    assert client.get("/api/status").status_code == 401
    assert client.get("/", params={"token": "guess"}).status_code == 401


def test_mutations_require_csrf_and_a_pending_intervention() -> None:
    client = _client()
    landing = client.get("/", params={"token": "local-token"})
    assert landing.status_code == 200
    csrf = re.search(r'const csrf = "([^"]+)"', landing.text)
    assert csrf is not None
    assert client.post("/api/claim").status_code == 403
    cross_origin = client.post("/api/claim", headers={"X-CSRF-Token": csrf.group(1), "Origin": "http://evil.example"})
    assert cross_origin.status_code == 403
    refused = client.post("/api/claim", headers={"X-CSRF-Token": csrf.group(1)})
    assert refused.status_code == 409
    assert client.get("/api/status").json()["control"]["state"] == "automation"
