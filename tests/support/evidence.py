from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class InMemoryEvidenceSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.images: list[str] = []
        self.documents: dict[str, dict[str, Any]] = {}

    def emit(self, event: str, data: Mapping[str, Any]) -> None:
        self.events.append((event, dict(data)))

    def attach_image(self, label: str, png: bytes) -> str:
        self.images.append(label)
        return f"screenshots/{len(self.images):04d}-{label}.png"

    def write_document(self, name: str, content: Mapping[str, Any]) -> str:
        self.documents[name] = dict(content)
        return f"{name}.json"

    def named(self, event: str) -> list[dict[str, Any]]:
        return [data for name, data in self.events if name == event]

    def serialized(self) -> str:
        return json.dumps({"events": self.events, "documents": self.documents}, default=str)
