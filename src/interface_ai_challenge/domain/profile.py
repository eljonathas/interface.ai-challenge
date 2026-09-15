from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from interface_ai_challenge.domain.artifact import SurfaceFeature
from interface_ai_challenge.domain.handlers import Handler
from interface_ai_challenge.domain.targets import NamedTarget, Target


class RedactionRules(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_screenshots: tuple[Target, ...] = ()
    evidence_screenshots: tuple[Target, ...] = ()


class TargetProfile(BaseModel):
    """Authored, reviewed knowledge about an application family: known exceptional states and redaction regions."""

    model_config = ConfigDict(frozen=True)

    product: str
    ui_family: str
    requires: tuple[SurfaceFeature, ...]
    targets: dict[str, NamedTarget]
    handlers: tuple[Handler, ...]
    redaction: RedactionRules = RedactionRules()
