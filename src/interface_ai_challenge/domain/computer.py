from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from interface_ai_challenge.domain.geometry import Point


class ClickCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["click"] = "click"
    point: Point
    button: Literal["left", "right", "middle"] = "left"


class TypeCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["type"] = "type"
    text: str


class KeypressCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["keypress"] = "keypress"
    keys: tuple[str, ...] = Field(min_length=1)


class ScrollCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["scroll"] = "scroll"
    point: Point
    delta_x: int
    delta_y: int


class WaitCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["wait"] = "wait"
    milliseconds: int = Field(default=1000, gt=0, le=5000)


class ScreenshotCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["screenshot"] = "screenshot"


class UnsupportedCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["unsupported"] = "unsupported"
    original_kind: str


ComputerCommand = Annotated[
    ClickCommand | TypeCommand | KeypressCommand | ScrollCommand | WaitCommand | ScreenshotCommand | UnsupportedCommand,
    Field(discriminator="kind"),
]
