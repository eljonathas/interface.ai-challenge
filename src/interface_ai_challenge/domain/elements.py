from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from interface_ai_challenge.domain.geometry import Rect
from interface_ai_challenge.domain.targets import FrameRef

ElementRole = Literal["button", "link", "textbox", "canvas", "cell", "text", "other"]


class ElementDescriptor(BaseModel):
    """What the surface observed at a point: visible, non-positional facts used to synthesize targets."""

    model_config = ConfigDict(frozen=True)

    frame: FrameRef
    frame_is_named: bool = True
    role: ElementRole
    name: str
    label: str | None = None
    row_label: str | None = None
    column_header: str | None = None
    row_cells: dict[str, str] = Field(default_factory=dict)
    rect: Rect

    @property
    def is_canvas(self) -> bool:
        return self.role == "canvas"
