from __future__ import annotations

import secrets
from enum import StrEnum

from interface_ai_challenge.demo_app.data import SYNTHETIC_MEMBERS, Member


class Scenario(StrEnum):
    NORMAL = "normal"
    MAINTENANCE = "maintenance"
    SLOW = "slow"
    TRANSIENT_ERROR = "transient_error"
    SESSION_EXPIRED = "session_expired"
    PERMISSION_DENIED = "permission_denied"
    UNKNOWN_DIALOG = "unknown_dialog"
    CANVAS_SHIFT = "canvas_shift"
    DUPLICATE_CANVAS_CONTROL = "duplicate_canvas_control"
    INJECTION = "injection"


class DemoState:
    """Server-side state of the stand-in application. Invisible to automation except through its UI."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self._consumed: set[str] = set()
        self._tokens: dict[str, str] = {}

    def first_time(self, flag: str) -> bool:
        if flag in self._consumed:
            return False
        self._consumed.add(flag)
        return True

    def token_for(self, member_id: str) -> str:
        token = secrets.token_hex(6)
        self._tokens[token] = member_id
        return token

    def member_for(self, token: str) -> Member | None:
        member_id = self._tokens.get(token)
        return SYNTHETIC_MEMBERS.get(member_id) if member_id else None
