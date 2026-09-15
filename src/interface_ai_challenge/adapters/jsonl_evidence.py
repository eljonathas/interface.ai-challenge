from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonlEvidenceSink:
    def __init__(
        self,
        directory: Path,
        run_id: str,
        mode: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._directory = directory
        self._run_id = run_id
        self._mode = mode
        self._clock = clock
        self._sequence = 0
        (directory / "screenshots").mkdir(parents=True, exist_ok=True)
        self._events = directory / "events.jsonl"

    @property
    def directory(self) -> Path:
        return self._directory

    def emit(self, event: str, data: Mapping[str, Any]) -> None:
        self._sequence += 1
        record = {
            "seq": self._sequence,
            "ts": self._clock().isoformat(),
            "run_id": self._run_id,
            "mode": self._mode,
            "event": event,
            "data": dict(data),
        }
        with self._events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str, sort_keys=True) + "\n")

    def attach_image(self, label: str, png: bytes) -> str:
        self._sequence += 1
        relative = Path("screenshots") / f"{self._sequence:04d}-{label}.png"
        (self._directory / relative).write_bytes(png)
        return relative.as_posix()

    def write_document(self, name: str, content: Mapping[str, Any]) -> str:
        path = self._directory / f"{name}.json"
        path.write_text(json.dumps(dict(content), indent=2, default=str, sort_keys=True) + "\n", encoding="utf-8")
        return path.name
