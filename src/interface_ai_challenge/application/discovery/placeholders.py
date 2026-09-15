from __future__ import annotations

import re

from interface_ai_challenge.domain.values import InputRef, ValueResolver

_PLACEHOLDER = re.compile(r"\{\{\s*inputs\.([a-z][a-z0-9_]*)\s*\}\}")


class PlaceholderError(ValueError):
    pass


class PlaceholderResolver:
    """Lets the model type `{{inputs.name}}` so parameter values are bound by reference, not guessed from text."""

    def __init__(self, values: ValueResolver) -> None:
        self._values = values

    def input_ref(self, text: str) -> InputRef | None:
        match = _PLACEHOLDER.fullmatch(text.strip())
        if match is None:
            if _PLACEHOLDER.search(text):
                raise PlaceholderError("Type an input placeholder on its own, for example {{inputs.member_id}}.")
            return None
        name = match.group(1)
        if name not in self._values.input_names():
            raise PlaceholderError(f"Unknown input placeholder '{name}'.")
        return InputRef(name=name)

    def resolve(self, text: str) -> str:
        ref = self.input_ref(text)
        return text if ref is None else self._values.resolve(ref)

    @staticmethod
    def placeholder_for(name: str) -> str:
        return "{{inputs." + name + "}}"
