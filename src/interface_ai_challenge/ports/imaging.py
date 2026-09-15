from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from interface_ai_challenge.domain.geometry import Rect


@dataclass(frozen=True)
class TemplateMatch:
    rect: Rect
    score: float


class TemplateMatcher(Protocol):
    def find(
        self, template_png: bytes, image_png: bytes, region: Rect | None, min_score: float
    ) -> list[TemplateMatch]: ...


class ImageEditor(Protocol):
    def crop(self, image_png: bytes, rect: Rect) -> bytes: ...

    def mask(self, image_png: bytes, rects: Sequence[Rect]) -> bytes: ...


class AssetStore(Protocol):
    def get(self, sha256: str) -> bytes: ...

    def put(self, content: bytes) -> str: ...
