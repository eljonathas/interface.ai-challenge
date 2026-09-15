from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from interface_ai_challenge.domain.errors import SurfaceError
from interface_ai_challenge.domain.geometry import Rect, Screenshot
from interface_ai_challenge.domain.targets import Target
from interface_ai_challenge.domain.values import ValueResolver
from interface_ai_challenge.ports.evidence import EvidenceSink
from interface_ai_challenge.ports.imaging import ImageEditor
from interface_ai_challenge.ports.surface import ScreenCapture, TargetLocator

_MIN_SECRET_LENGTH = 3


class Redactor:
    """Last line of defence before persistence: replaces known sensitive values wherever they appear."""

    def __init__(self) -> None:
        self._secrets: dict[str, str] = {}

    def register(self, label: str, value: str) -> None:
        if len(value) >= _MIN_SECRET_LENGTH:
            self._secrets[value] = label

    def redact(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._redact_text(value)
        if isinstance(value, Mapping):
            return {key: self.redact(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [self.redact(item) for item in value]
        return value

    def _redact_text(self, text: str) -> str:
        for secret in sorted(self._secrets, key=len, reverse=True):
            text = text.replace(secret, f"[REDACTED:{self._secrets[secret]}]")
        return text


class RedactingEvidenceSink:
    def __init__(self, inner: EvidenceSink, redactor: Redactor) -> None:
        self._inner = inner
        self._redactor = redactor

    def emit(self, event: str, data: Mapping[str, Any]) -> None:
        self._inner.emit(event, self._redactor.redact(data))

    def attach_image(self, label: str, png: bytes) -> str:
        return self._inner.attach_image(label, png)

    def write_document(self, name: str, content: Mapping[str, Any]) -> str:
        return self._inner.write_document(name, self._redactor.redact(content))


class MaskedScreenCapture:
    """Screenshots with sensitive regions blacked out. Fails closed: if masking fails, no image is produced."""

    def __init__(
        self,
        screen: ScreenCapture,
        locator: TargetLocator,
        editor: ImageEditor,
        regions: Sequence[Target],
        values: ValueResolver,
    ) -> None:
        self._screen = screen
        self._locator = locator
        self._editor = editor
        self._regions = tuple(regions)
        self._values = values

    def viewport(self) -> Any:
        return self._screen.viewport()

    async def capture(self) -> Screenshot:
        screenshot = await self._screen.capture()
        rects: list[Rect] = []
        for region in self._regions:
            try:
                candidates = await self._locator.locate(region, self._values)
            except SurfaceError:
                raise SurfaceError("screenshot redaction could not be completed") from None
            rects.extend(candidate.rect for candidate in candidates)
        if not rects:
            return screenshot
        return Screenshot(
            png=self._editor.mask(screenshot.png, rects),
            size=screenshot.size,
            revision=screenshot.revision,
        )
