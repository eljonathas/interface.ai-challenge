from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any, TextIO

from interface_ai_challenge.ports.evidence import EvidenceSink
from interface_ai_challenge.ports.model import AgentTurn


class ConsoleEvidenceSink:
    """Mirrors evidence events to the terminal while a run is live. Wrap it in the redacting sink."""

    def __init__(self, inner: EvidenceSink, mode: str, stream: TextIO | None = None) -> None:
        self._inner = inner
        self._mode = mode
        self._stream = stream

    def emit(self, event: str, data: Mapping[str, Any]) -> None:
        self._inner.emit(event, data)
        self._print(f"[{self._mode}] {event} {json.dumps(dict(data), default=str, sort_keys=True)}")

    def attach_image(self, label: str, png: bytes) -> str:
        path = self._inner.attach_image(label, png)
        self._print(f"[{self._mode}] screenshot {path}")
        return path

    def write_document(self, name: str, content: Mapping[str, Any]) -> str:
        return self._inner.write_document(name, content)

    def _print(self, line: str) -> None:
        print(line, file=self._stream or sys.stderr, flush=True)


def print_model_turn(turn: AgentTurn, stream: TextIO | None = None) -> None:
    """Live view of one model decision. Free model text is shown here only and is never persisted."""
    lines = [f"[model] response {turn.request_id or '-'}"]
    lines += [f"  reasoning: {text}" for text in turn.reasoning]
    if turn.message:
        lines.append(f"  message: {turn.message}")
    lines += [f"  action: {json.dumps(command.model_dump(mode='json'))}" for command in turn.commands]
    lines += [f"  signal: {signal}" for signal in turn.signals]
    lines += [f"  safety check: {check}" for check in turn.safety_checks]
    if turn.observation_required:
        lines.append("  (screenshot round trip: nothing from this turn is executed)")
    print("\n".join(lines), file=stream or sys.stderr, flush=True)
