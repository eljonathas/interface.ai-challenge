import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from interface_ai_challenge.adapters.file_store import load_model
from interface_ai_challenge.adapters.openai_agent import (
    OpenAIAgentSettings,
    OpenAIComputerUseAgent,
    OpenAIGoalInterpreter,
)
from interface_ai_challenge.agents import create_agent
from interface_ai_challenge.domain.computer import ClickCommand, KeypressCommand, TypeCommand, UnsupportedCommand
from interface_ai_challenge.domain.geometry import Point, Screenshot, Size
from interface_ai_challenge.ports.model import AgentFeedback, AgentTask, OutputSpec, ReportOutputSignal
from interface_ai_challenge.settings import ModelSettings, SystemConfig

SCREENSHOT = Screenshot(png=b"png-bytes", size=Size(width=1280, height=800), revision=1)
TASK = AgentTask(
    goal="Look up the member {{inputs.member_id}}",
    outputs=(OutputSpec(name="balance", description="Current balance"),),
    input_placeholders=("{{inputs.member_id}}",),
)


class FakeResponses:
    def __init__(self, responses: list[Any]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._responses = responses

    async def create(self, **request: Any) -> Any:
        if "previous_response_id" in request:
            # Reproduce the restriction confirmed against the live computer API.
            assert not any(
                part.get("type") == "input_image" for item in request["input"] for part in item.get("content", [])
            )
        self.requests.append(request)
        return self._responses.pop(0)


def _usage() -> SimpleNamespace:
    return SimpleNamespace(input_tokens=1500, output_tokens=40, input_tokens_details=SimpleNamespace(cached_tokens=500))


def _responses() -> list[Any]:
    computer_call = SimpleNamespace(
        type="computer_call",
        call_id="call_1",
        action=None,
        actions=[
            SimpleNamespace(type="click", button="left", x=410, y=220),
            SimpleNamespace(type="type", text="{{inputs.member_id}}"),
            SimpleNamespace(type="keypress", keys=["ENTER"]),
            SimpleNamespace(type="drag", path=[]),
        ],
        pending_safety_checks=[],
    )
    report = SimpleNamespace(
        type="function_call",
        call_id="call_2",
        name="report_output",
        arguments='{"output": "balance", "x": 700, "y": 400}',
    )
    message = SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text="Balance is visible.")])
    thought = SimpleNamespace(type="reasoning", summary=[SimpleNamespace(type="summary_text", text="Found the row.")])
    return [
        SimpleNamespace(id="resp_1", output=[computer_call], usage=_usage()),
        SimpleNamespace(id="resp_2", output=[thought, message, report], usage=_usage()),
        SimpleNamespace(id="resp_3", output=[], usage=_usage()),
    ]


async def test_adapter_speaks_responses_api_and_returns_neutral_turns() -> None:
    responses = FakeResponses(_responses())
    settings = OpenAIAgentSettings(model="gpt-5.4", reasoning_summary="auto")
    agent = OpenAIComputerUseAgent(SimpleNamespace(responses=responses), settings)  # type: ignore[arg-type]

    first = await agent.begin(TASK, SCREENSHOT)
    request = responses.requests[0]
    assert request["model"] == "gpt-5.4"
    assert request["reasoning"] == {"effort": "low", "summary": "auto"}
    assert request["tools"][0] == {"type": "computer"}
    assert {tool["name"] for tool in request["tools"][1:]} == {
        "report_output",
        "report_identity",
        "finish",
        "request_help",
    }
    image = request["input"][0]["content"][1]
    assert image["detail"] == "original" and image["image_url"].startswith("data:image/png;base64,")
    assert "previous_response_id" not in request
    assert first.commands == (
        ClickCommand(point=Point(x=410, y=220)),
        TypeCommand(text="{{inputs.member_id}}"),
        KeypressCommand(keys=("Enter",)),
        UnsupportedCommand(original_kind="drag"),
    )
    assert first.usage.cached_input_tokens == 500

    second = await agent.respond(AgentFeedback(screenshot=SCREENSHOT, notes=["Action blocked by policy: drag"]))
    request = responses.requests[1]
    assert request["previous_response_id"] == "resp_1"
    assert request["input"][0]["type"] == "computer_call_output"
    assert request["input"][0]["call_id"] == "call_1"
    assert request["input"][1]["content"][0]["text"] == "Action blocked by policy: drag"
    assert second.signals == (ReportOutputSignal(call_id="call_2", output="balance", point=Point(x=700, y=400)),)
    assert second.message == "Balance is visible." and second.reasoning == ("Found the row.",)

    await agent.respond(AgentFeedback(screenshot=SCREENSHOT, signal_replies={"call_2": "Recorded."}))
    request = responses.requests[2]
    outputs = [item for item in request["input"] if item.get("type") == "function_call_output"]
    assert outputs == [{"type": "function_call_output", "call_id": "call_2", "output": "Recorded."}]
    assert request["tool_choice"] == {"type": "computer"}


async def test_help_reply_refreshes_through_the_computer_protocol_before_acting() -> None:
    responses = FakeResponses(
        [
            SimpleNamespace(
                id="help-response",
                output=[
                    SimpleNamespace(
                        type="function_call", call_id="help", name="request_help", arguments='{"reason":"login"}'
                    )
                ],
                usage=_usage(),
            ),
            SimpleNamespace(
                id="screenshot-response",
                output=[
                    SimpleNamespace(
                        type="computer_call",
                        call_id="screen",
                        action=None,
                        actions=[SimpleNamespace(type="screenshot")],
                        pending_safety_checks=[],
                    )
                ],
                usage=_usage(),
            ),
            SimpleNamespace(id="next-response", output=[], usage=_usage()),
        ]
    )
    agent = OpenAIComputerUseAgent(SimpleNamespace(responses=responses), OpenAIAgentSettings(model="gpt-5.4"))  # type: ignore[arg-type]
    await agent.begin(TASK, SCREENSHOT)
    fresh = Screenshot(png=b"new-screen-after-human", size=SCREENSHOT.size, revision=2)
    refresh = await agent.respond(AgentFeedback(fresh, {"help": "Human returned control."}, ["Decide again."]))
    assert refresh.observation_required
    assert responses.requests[-1]["previous_response_id"] == "help-response"
    assert responses.requests[-1]["tool_choice"] == {"type": "computer"}
    message = responses.requests[-1]["input"][-1]
    assert message["content"][0]["text"].startswith("Decide again.")
    next_turn = await agent.respond(AgentFeedback(fresh))
    assert not next_turn.observation_required
    request = responses.requests[-1]
    assert request["previous_response_id"] == "screenshot-response"
    assert "tool_choice" not in request
    assert request["input"][0]["call_id"] == "screen"
    image = request["input"][0]["output"]
    assert image["type"] == "computer_screenshot"
    assert base64.b64decode(image["image_url"].split(",", 1)[1]) == fresh.png


async def test_goal_interpreter_returns_a_structured_proposal() -> None:
    arguments = {
        "capability_id": "member.savings_balance",
        "description": "Reads a member's savings balance.",
        "parameters": [
            {"name": "member_id", "description": "Member ID", "value": "M-10023", "identifies_record": True}
        ],
        "outputs": [{"name": "balance", "description": "Savings balance", "kind": "decimal_money"}],
        "clarification": None,
    }
    call = SimpleNamespace(type="function_call", call_id="c", name="define_capability", arguments=json.dumps(arguments))
    responses = FakeResponses([SimpleNamespace(id="r", output=[call], usage=_usage())])
    interpreter = OpenAIGoalInterpreter(SimpleNamespace(responses=responses), OpenAIAgentSettings(model="gpt-5.4"))  # type: ignore[arg-type]
    interpretation = await interpreter.interpret("look up member M-10023 and read their savings balance")
    assert responses.requests[0]["tool_choice"] == {"type": "function", "name": "define_capability"}
    assert interpretation.parameters[0].value == "M-10023" and interpretation.parameters[0].identifies_record
    assert interpretation.outputs[0].kind == "decimal_money" and interpretation.clarification is None


def test_model_and_provider_are_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    load_model(Path(__file__).resolve().parents[2] / "configs" / "system.json", SystemConfig)
    config = SystemConfig.model_validate({"llm": {"model": "gpt-5.4", "reasoning_effort": None}})
    settings = config.llm_settings()
    assert settings.model == "gpt-5.4" and settings.reasoning_effort is None
    assert config.llm_settings(model="gpt-5.6-terra").model == "gpt-5.6-terra"
    assert config.runtime().headless and not config.runtime().enable_operator
    assert config.runtime(8765).enable_operator
    with pytest.raises(ValueError, match="unknown provider"):
        create_agent(ModelSettings(provider="anthropic"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    agent = create_agent(ModelSettings(model="gpt-5.4"))
    assert (agent.provider, agent.model) == ("openai", "gpt-5.4")
