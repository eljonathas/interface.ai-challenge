from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from interface_ai_challenge.domain.geometry import Rect
from interface_ai_challenge.ports.imaging import TemplateMatch

_MAX_MATCHES = 8
_MIN_TEMPLATE_CONTRAST = 4.0


def _decode(png: bytes, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), flags)
    if image is None:
        raise ValueError("invalid PNG data")
    return image


def _encode(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("could not encode PNG")
    return buffer.tobytes()


class OpenCvTemplateMatcher:
    """Normalized cross-correlation with non-maximum suppression, so one visual occurrence counts once."""

    def find(self, template_png: bytes, image_png: bytes, region: Rect | None, min_score: float) -> list[TemplateMatch]:
        image = _decode(image_png)
        template = _decode(template_png)
        offset_x = offset_y = 0
        if region is not None:
            offset_x, offset_y = int(region.x), int(region.y)
            image = image[offset_y : offset_y + int(region.height), offset_x : offset_x + int(region.width)]
        height, width = template.shape
        if image.shape[0] < height or image.shape[1] < width or float(template.std()) < _MIN_TEMPLATE_CONTRAST:
            return []
        scores = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        matches: list[TemplateMatch] = []
        for _ in range(_MAX_MATCHES):
            _, score, _, (x, y) = cv2.minMaxLoc(scores)
            if score < min_score:
                break
            matches.append(
                TemplateMatch(rect=Rect(x=x + offset_x, y=y + offset_y, width=width, height=height), score=float(score))
            )
            scores[max(0, y - height // 2) : y + height // 2 + 1, max(0, x - width // 2) : x + width // 2 + 1] = -1.0
        return matches


class OpenCvImageEditor:
    def crop(self, image_png: bytes, rect: Rect) -> bytes:
        image = _decode(image_png, cv2.IMREAD_COLOR)
        x, y = int(rect.x), int(rect.y)
        return _encode(image[y : y + int(rect.height), x : x + int(rect.width)])

    def mask(self, image_png: bytes, rects: Sequence[Rect]) -> bytes:
        image = _decode(image_png, cv2.IMREAD_COLOR)
        for rect in rects:
            top_left = (int(rect.x) - 2, int(rect.y) - 2)
            bottom_right = (int(rect.x + rect.width) + 2, int(rect.y + rect.height) + 2)
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), thickness=-1)
        return _encode(image)
