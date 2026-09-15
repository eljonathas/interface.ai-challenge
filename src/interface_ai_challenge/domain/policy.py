from __future__ import annotations

from fnmatch import fnmatchcase
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from interface_ai_challenge.domain.control import ActorKind
from interface_ai_challenge.domain.steps import Effect

_DEFAULT_PORTS = {"http": 80, "https": 443}


class WebOrigin(BaseModel):
    model_config = ConfigDict(frozen=True)

    scheme: str
    host: str
    port: int

    @classmethod
    def parse(cls, url: str) -> WebOrigin | None:
        parts = urlsplit(url)
        if parts.scheme not in _DEFAULT_PORTS or not parts.hostname:
            return None
        try:
            port = parts.port or _DEFAULT_PORTS[parts.scheme]
        except ValueError:
            return None
        return cls(scheme=parts.scheme, host=parts.hostname.lower(), port=port)


class PolicyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    allowed_origins: tuple[str, ...] = Field(min_length=1)
    allowed_paths: tuple[str, ...] = Field(min_length=1)
    allowed_commands: tuple[str, ...]
    allowed_keys: tuple[str, ...]
    irreversible_control_names: tuple[str, ...] = ()
    canvas_click_effect: Effect = Effect.UNKNOWN
    automation_effects: tuple[Effect, ...]
    human_effects: tuple[Effect, ...]
    max_typed_length: int = Field(default=256, gt=0)

    def effects_for(self, actor: ActorKind) -> tuple[Effect, ...]:
        return self.automation_effects if actor is ActorKind.AUTOMATION else self.human_effects


class UrlPolicy:
    def __init__(self, config: PolicyConfig) -> None:
        self._origins = {WebOrigin.parse(origin) for origin in config.allowed_origins}
        self._paths = config.allowed_paths

    def allows(self, url: str) -> bool:
        if url == "about:blank":
            return True
        origin = WebOrigin.parse(url)
        if origin is None or origin not in self._origins:
            return False
        path = _normalize_path(urlsplit(url).path)
        return path is not None and any(fnmatchcase(path, pattern) for pattern in self._paths)


def _normalize_path(path: str) -> str | None:
    segments: list[str] = []
    for segment in (path or "/").split("/"):
        if segment in ("", "."):
            continue
        if segment == ".." or "%2e" in segment.lower() or "%2f" in segment.lower():
            return None
        segments.append(segment)
    return "/" + "/".join(segments)
