from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from interface_ai_challenge.domain.errors import ExtractionError, InputValidationError
from interface_ai_challenge.domain.targets import TableCellLocator, Target, WebTarget

_IDENTIFIER = r"^[a-z][a-z0-9_]*$"


class InputField(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=_IDENTIFIER)
    type: Literal["string"] = "string"
    description: str
    sensitive: bool = True
    max_length: int = Field(default=64, gt=0)
    # Discovery must point at this value on the opened record; replay checks it before returning outputs.
    identifies_record: bool = False

    def validate_value(self, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InputValidationError(self.name, "must be a non-empty string")
        if len(value) > self.max_length:
            raise InputValidationError(self.name, f"must be at most {self.max_length} characters")
        return value


class TextParser(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["text"] = "text"

    def parse(self, raw: str) -> str:
        return raw.strip()


class EnumParser(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["enum"] = "enum"
    mapping: dict[str, str] = Field(min_length=1)

    def parse(self, raw: str) -> str:
        key = raw.strip()
        if key not in self.mapping:
            raise ExtractionError(f"value is not one of {sorted(self.mapping)}")
        return self.mapping[key]


class DecimalMoneyParser(BaseModel):
    """Parses en-US money text such as `$1,250.45` into a two-decimal string."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["decimal_money"] = "decimal_money"
    locale: Literal["en-US"] = "en-US"

    def parse(self, raw: str) -> str:
        cleaned = raw.strip().removeprefix("$").replace(",", "")
        if not re.fullmatch(r"-?\d+(\.\d{1,2})?", cleaned):
            raise ExtractionError("value is not an en-US monetary amount")
        try:
            return str(Decimal(cleaned).quantize(Decimal("0.01")))
        except InvalidOperation as error:
            raise ExtractionError("value is not a decimal amount") from error


Parser = Annotated[TextParser | EnumParser | DecimalMoneyParser, Field(discriminator="kind")]


class OutputField(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=_IDENTIFIER)
    description: str
    parser: Parser
    sensitive: bool = False


class CapabilityContract(BaseModel):
    model_config = ConfigDict(frozen=True)

    capability_id: str = Field(pattern=r"^[a-z][a-z0-9_.]*$")
    description: str
    goal: str
    inputs: tuple[InputField, ...] = Field(min_length=1)
    outputs: tuple[OutputField, ...] = Field(min_length=1)
    business_outcomes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _goal_uses_placeholders(self) -> CapabilityContract:
        for field in self.inputs:
            if "{{inputs." + field.name + "}}" not in self.goal:
                raise ValueError(f"goal must reference input '{field.name}' as a placeholder")
        return self

    def output_named(self, name: str) -> OutputField:
        return next(field for field in self.outputs if field.name == name)

    def validate_inputs(self, values: dict[str, object]) -> dict[str, str]:
        unknown = set(values) - {field.name for field in self.inputs}
        if unknown:
            raise InputValidationError(sorted(unknown)[0], "is not declared by the contract")
        validated: dict[str, str] = {}
        for field in self.inputs:
            if field.name not in values:
                raise InputValidationError(field.name, "is required")
            validated[field.name] = field.validate_value(values[field.name])
        return validated


def output_row_problems(targets: Mapping[str, Sequence[Target]]) -> list[str]:
    """Table-cell outputs in one frame describe one record, so every strategy must keep the same row identity."""
    rows: dict[tuple[str, ...], dict[str, set[str]]] = {}
    for name, strategies in targets.items():
        for target in strategies:
            if isinstance(target, WebTarget) and isinstance(target.locator, TableCellLocator):
                locator = target.locator
                scope = f"{locator.row_key_column}={locator.row_key.model_dump_json()}"
                rows.setdefault(target.frame.names, {}).setdefault(scope, set()).add(name)
    return [
        f"outputs {sorted(set().union(*scopes.values()))} must come from the same table row"
        for scopes in rows.values()
        if len(scopes) > 1
    ]
