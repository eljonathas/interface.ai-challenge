from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


class EvidenceSink(Protocol):
    def emit(self, event: str, data: Mapping[str, Any]) -> None: ...

    def attach_image(self, label: str, png: bytes) -> str: ...

    def write_document(self, name: str, content: Mapping[str, Any]) -> str: ...
