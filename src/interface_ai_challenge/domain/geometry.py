from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


class Point(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: int
    y: int


class Size(BaseModel):
    model_config = ConfigDict(frozen=True)

    width: int
    height: int


class Rect(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    width: float
    height: float

    def center(self) -> Point:
        return Point(x=round(self.x + self.width / 2), y=round(self.y + self.height / 2))

    def contains(self, point: Point) -> bool:
        return self.x <= point.x <= self.x + self.width and self.y <= point.y <= self.y + self.height


@dataclass(frozen=True)
class Screenshot:
    png: bytes
    size: Size
    revision: int
