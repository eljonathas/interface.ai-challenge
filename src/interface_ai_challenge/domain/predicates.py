from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from interface_ai_challenge.domain.targets import FrameRef
from interface_ai_challenge.domain.values import ValueRef


class Visible(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["visible"] = "visible"
    target_ref: str


class Absent(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["absent"] = "absent"
    target_ref: str


class TextEquals(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["text_equals"] = "text_equals"
    target_ref: str
    value: ValueRef


class FieldEquals(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["field_equals"] = "field_equals"
    target_ref: str
    value: ValueRef


class TextVisible(BaseModel):
    """Visible text anywhere in the given frame, or in any frame when `frame` is None."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["text_visible"] = "text_visible"
    text: str
    frame: FrameRef | None = None


class DialogVisible(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["dialog_visible"] = "dialog_visible"


class AllOf(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["all"] = "all"
    predicates: tuple[Predicate, ...] = Field(min_length=1)


class AnyOf(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["any"] = "any"
    predicates: tuple[Predicate, ...] = Field(min_length=1)


Predicate = Annotated[
    Visible | Absent | TextEquals | FieldEquals | TextVisible | DialogVisible | AllOf | AnyOf,
    Field(discriminator="kind"),
]

AllOf.model_rebuild()
AnyOf.model_rebuild()


def target_refs(predicate: Predicate) -> set[str]:
    if isinstance(predicate, AllOf | AnyOf):
        return set().union(*(target_refs(child) for child in predicate.predicates))
    if isinstance(predicate, Visible | Absent | TextEquals | FieldEquals):
        return {predicate.target_ref}
    return set()


def value_refs(predicate: Predicate) -> list[ValueRef]:
    if isinstance(predicate, AllOf | AnyOf):
        return [ref for child in predicate.predicates for ref in value_refs(child)]
    if isinstance(predicate, TextEquals | FieldEquals):
        return [predicate.value]
    return []
