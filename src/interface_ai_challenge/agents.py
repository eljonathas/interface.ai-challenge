from __future__ import annotations

from collections.abc import Callable

from interface_ai_challenge.ports.model import ComputerUseAgent, GoalInterpreter
from interface_ai_challenge.settings import ModelSettings

AgentFactory = Callable[[ModelSettings], ComputerUseAgent]


def _openai(settings: ModelSettings) -> ComputerUseAgent:
    from openai import AsyncOpenAI

    from interface_ai_challenge.adapters.openai_agent import OpenAIAgentSettings, OpenAIComputerUseAgent

    return OpenAIComputerUseAgent(
        AsyncOpenAI(),
        OpenAIAgentSettings(
            model=settings.model,
            reasoning_effort=settings.reasoning_effort,
            reasoning_summary=settings.reasoning_summary,
        ),
    )


def _openai_interpreter(settings: ModelSettings) -> GoalInterpreter:
    from openai import AsyncOpenAI

    from interface_ai_challenge.adapters.openai_agent import OpenAIAgentSettings, OpenAIGoalInterpreter

    return OpenAIGoalInterpreter(
        AsyncOpenAI(),
        OpenAIAgentSettings(
            model=settings.model,
            reasoning_effort=settings.reasoning_effort,
            reasoning_summary=settings.reasoning_summary,
        ),
    )


AGENT_FACTORIES: dict[str, AgentFactory] = {"openai": _openai}
INTERPRETER_FACTORIES: dict[str, Callable[[ModelSettings], GoalInterpreter]] = {"openai": _openai_interpreter}

PROVIDER_CREDENTIALS: dict[str, str] = {"openai": "OPENAI_API_KEY"}


def create_agent(settings: ModelSettings) -> ComputerUseAgent:
    """Provider SDKs are imported lazily so replay never needs them installed."""

    return _factory(AGENT_FACTORIES, settings)(settings)


def create_goal_interpreter(settings: ModelSettings) -> GoalInterpreter:
    return _factory(INTERPRETER_FACTORIES, settings)(settings)


def _factory[T](
    factories: dict[str, Callable[[ModelSettings], T]], settings: ModelSettings
) -> Callable[[ModelSettings], T]:
    factory = factories.get(settings.provider)
    if factory is None:
        raise ValueError(f"unknown provider '{settings.provider}'; available: {sorted(factories)}")
    return factory
