from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from interface_ai_challenge.domain.geometry import Point, Rect, Size
from interface_ai_challenge.domain.values import ValueRef

ControlRole = Literal["button", "link", "textbox"]


class FrameRef(BaseModel):
    """Path of iframe `name` attributes from the top document; empty means the top document."""

    model_config = ConfigDict(frozen=True)

    names: tuple[str, ...] = ()


class RoleLocator(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: Literal["role"] = "role"
    role: ControlRole
    name: str


class LabeledFieldLocator(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: Literal["labeled_field"] = "labeled_field"
    label: str


class LabeledValueLocator(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: Literal["labeled_value"] = "labeled_value"
    label: str


class TableCellLocator(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: Literal["table_cell"] = "table_cell"
    column: str
    row_key_column: str
    row_key: ValueRef
    control: RoleLocator | None = None


class TableColumnLocator(BaseModel):
    """Every data cell under a column header, including rows unrelated to the requested member."""

    model_config = ConfigDict(frozen=True)

    strategy: Literal["table_column"] = "table_column"
    column: str


WebLocator = Annotated[
    RoleLocator | LabeledFieldLocator | LabeledValueLocator | TableCellLocator | TableColumnLocator,
    Field(discriminator="strategy"),
]


class WebTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["web"] = "web"
    frame: FrameRef = FrameRef()
    locator: WebLocator


class VisualTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["visual"] = "visual"
    template_sha256: str
    template_size: Size
    click_offset: Point
    search_region: Rect | None = None
    viewport: Size
    min_score: float = Field(ge=0.5, le=1.0, default=0.92)


Target = Annotated[WebTarget | VisualTarget, Field(discriminator="kind")]


class NamedTarget(BaseModel):
    """Ordered strategies: a later one is tried only when every earlier one matched nothing."""

    model_config = ConfigDict(frozen=True)

    strategies: tuple[Target, ...] = Field(min_length=1)
    rationale: str
