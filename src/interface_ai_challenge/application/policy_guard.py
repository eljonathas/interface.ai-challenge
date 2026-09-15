from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from interface_ai_challenge.domain.control import ActorKind
from interface_ai_challenge.domain.elements import ElementDescriptor
from interface_ai_challenge.domain.policy import PolicyConfig, UrlPolicy
from interface_ai_challenge.domain.steps import Effect

_RISK_ORDER = (
    Effect.READ,
    Effect.NAVIGATION,
    Effect.REVERSIBLE_INPUT,
    Effect.SUBMIT,
    Effect.UNKNOWN,
    Effect.IRREVERSIBLE,
)
_FIELD_EDITING_KEYS = frozenset({"Backspace", "Delete", "ArrowLeft", "ArrowRight", "Home", "End", "Tab"})


class ActionIntent(BaseModel):
    model_config = ConfigDict(frozen=True)

    command: str
    effect: Effect
    control_name: str | None = None
    keys: tuple[str, ...] = ()
    url: str | None = None
    text_length: int | None = None


class PolicyDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str


def riskiest(*effects: Effect) -> Effect:
    return max(effects, key=_RISK_ORDER.index)


class PolicyGuard:
    def __init__(self, config: PolicyConfig) -> None:
        self._config = config
        self._urls = UrlPolicy(config)

    def evaluate(self, intent: ActionIntent, actor: ActorKind) -> PolicyDecision:
        config = self._config
        if intent.command not in config.allowed_commands:
            return _deny(f"command '{intent.command}' is not allowlisted")
        if intent.url is not None and not self._urls.allows(intent.url):
            return _deny("URL is outside the allowlisted origins/paths")
        disallowed_keys = [key for key in intent.keys if key not in config.allowed_keys]
        if disallowed_keys:
            return _deny(f"keys {disallowed_keys} are not allowlisted")
        if intent.text_length is not None and intent.text_length > config.max_typed_length:
            return _deny("typed text exceeds the allowed length")
        if intent.effect not in config.effects_for(actor):
            return _deny(f"effect '{intent.effect}' is not permitted for {actor}")
        return PolicyDecision(allowed=True, reason="allowlisted")

    def allows_url(self, url: str) -> bool:
        return self._urls.allows(url)


class EffectClassifier:
    """Classifies what an action may cause. Unknown stays unknown: the classifier never guesses 'safe'."""

    def __init__(self, config: PolicyConfig) -> None:
        self._config = config

    def for_click(self, descriptor: ElementDescriptor | None) -> Effect:
        if descriptor is None:
            return Effect.UNKNOWN
        return self.for_control(descriptor.role, descriptor.name)

    def for_control(self, role: str, name: str | None) -> Effect:
        if name and self._is_irreversible(name):
            return Effect.IRREVERSIBLE
        if role == "canvas":
            return self._config.canvas_click_effect
        if role == "textbox":
            return Effect.REVERSIBLE_INPUT
        if role == "link":
            return Effect.NAVIGATION
        if role == "button":
            return Effect.SUBMIT
        if role in ("cell", "text"):
            return Effect.READ
        return Effect.UNKNOWN

    def for_keys(self, keys: tuple[str, ...], focused: ElementDescriptor | None) -> Effect:
        if "Enter" in keys:
            if focused is None:
                return Effect.UNKNOWN
            return riskiest(Effect.SUBMIT, self.for_click(focused))
        if all(key in _FIELD_EDITING_KEYS for key in keys):
            return Effect.REVERSIBLE_INPUT
        return Effect.UNKNOWN

    def for_typing(self, focused: ElementDescriptor | None) -> Effect:
        if focused is None or focused.role != "textbox":
            return Effect.UNKNOWN
        return Effect.REVERSIBLE_INPUT

    def _is_irreversible(self, name: str) -> bool:
        folded = name.casefold()
        return any(term.casefold() in folded for term in self._config.irreversible_control_names)


def _deny(reason: str) -> PolicyDecision:
    return PolicyDecision(allowed=False, reason=reason)
