from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class InputRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["input"] = "input"
    name: str


class BindingRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["binding"] = "binding"
    name: str


class LiteralValue(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["literal"] = "literal"
    value: str


ValueRef = Annotated[InputRef | BindingRef | LiteralValue, Field(discriminator="kind")]


class UnknownValueError(KeyError):
    pass


class ValueResolver:
    def __init__(self, inputs: Mapping[str, str], bindings: Mapping[str, str]) -> None:
        self._inputs = dict(inputs)
        self._bindings = dict(bindings)

    def resolve(self, ref: InputRef | BindingRef | LiteralValue) -> str:
        if isinstance(ref, LiteralValue):
            return ref.value
        source = self._inputs if isinstance(ref, InputRef) else self._bindings
        if ref.name not in source:
            raise UnknownValueError(f"{ref.kind} '{ref.name}' was not supplied")
        return source[ref.name]

    def input_names(self) -> set[str]:
        return set(self._inputs)

    def input_matching(self, text: str) -> str | None:
        return next((name for name, value in self._inputs.items() if value == text), None)
