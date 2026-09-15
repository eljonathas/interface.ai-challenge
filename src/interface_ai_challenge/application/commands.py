from __future__ import annotations

from dataclasses import dataclass

from interface_ai_challenge.application.guarded_executor import GuardedExecutor
from interface_ai_challenge.application.policy_guard import ActionIntent, EffectClassifier
from interface_ai_challenge.domain.computer import (
    ClickCommand,
    ComputerCommand,
    KeypressCommand,
    ScreenshotCommand,
    ScrollCommand,
    TypeCommand,
    UnsupportedCommand,
    WaitCommand,
)
from interface_ai_challenge.domain.control import ControlToken
from interface_ai_challenge.domain.elements import ElementDescriptor
from interface_ai_challenge.domain.steps import Effect
from interface_ai_challenge.ports.surface import Surface


@dataclass(frozen=True)
class CommandReport:
    command: str
    effect: Effect
    descriptor: ElementDescriptor | None


class ComputerCommandExecutor:
    """Executes coordinate-level commands (from a model or an operator) through the guarded path."""

    def __init__(self, surface: Surface, executor: GuardedExecutor, classifier: EffectClassifier) -> None:
        self._surface = surface
        self._executor = executor
        self._classifier = classifier

    async def execute(self, token: ControlToken, command: ComputerCommand, text: str | None = None) -> CommandReport:
        raw = self._surface.raw_input
        inspector = self._surface.inspector
        match command:
            case ClickCommand(point=point, button=button):
                descriptor = await inspector.describe_at(point)
                effect = self._classifier.for_click(descriptor)
                name = descriptor.name if descriptor else None
                intent = ActionIntent(command="click", effect=effect, control_name=name)
                await self._executor.run(token, intent, lambda: raw.click(point, button))
            case TypeCommand(text=typed):
                descriptor = await inspector.describe_focused()
                resolved = typed if text is None else text
                effect = self._classifier.for_typing(descriptor)
                intent = ActionIntent(command="type", effect=effect, text_length=len(resolved))
                await self._executor.run(token, intent, lambda: raw.type_text(resolved))
            case KeypressCommand(keys=keys):
                descriptor = await inspector.describe_focused()
                effect = self._classifier.for_keys(keys, descriptor)
                intent = ActionIntent(command="keypress", effect=effect, keys=keys)
                await self._executor.run(token, intent, lambda: raw.press_keys(keys))
            case ScrollCommand(point=point, delta_x=dx, delta_y=dy):
                descriptor, effect = None, Effect.READ
                intent = ActionIntent(command="scroll", effect=effect)
                await self._executor.run(token, intent, lambda: raw.scroll(point, dx, dy))
            case WaitCommand(milliseconds=milliseconds):
                descriptor, effect = None, Effect.READ
                intent = ActionIntent(command="wait", effect=effect)
                await self._executor.run(token, intent, lambda: raw.wait(milliseconds))
            case ScreenshotCommand():
                return CommandReport(command="screenshot", effect=Effect.READ, descriptor=None)
            case UnsupportedCommand(original_kind=kind):
                descriptor, effect = None, Effect.UNKNOWN
                intent = ActionIntent(command=kind, effect=effect)
                await self._executor.run(token, intent, lambda: raw.wait(0))
        return CommandReport(command=command.kind, effect=effect, descriptor=descriptor)

    async def navigate(self, token: ControlToken, url: str) -> None:
        intent = ActionIntent(command="navigate", effect=Effect.NAVIGATION, url=url)
        await self._executor.run(token, intent, lambda: self._surface.navigator.goto(url))
