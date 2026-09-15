from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from interface_ai_challenge.domain.contract import CapabilityContract, output_row_problems
from interface_ai_challenge.domain.handlers import BusinessOutcomeResponse, Handler, RecoverByClickResponse
from interface_ai_challenge.domain.predicates import Predicate, target_refs, value_refs
from interface_ai_challenge.domain.steps import FillAction, NavigateAction, Step
from interface_ai_challenge.domain.targets import NamedTarget, TableCellLocator, VisualTarget, WebTarget
from interface_ai_challenge.domain.values import BindingRef, InputRef

SCHEMA_VERSION = "2.0"
SUPPORTED_SCHEMA_MAJOR = 2

SurfaceFeature = Literal["web.frames", "web.tables", "visual.template"]


class ApplicationRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    product: str
    ui_family: str
    requires: tuple[SurfaceFeature, ...]


class Entry(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str = Field(pattern=r"^/")
    required_bindings: tuple[str, ...] = ("base_url",)


class Extraction(BaseModel):
    model_config = ConfigDict(frozen=True)

    output: str
    target_ref: str


class Provenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    discovery_run_id: str
    provider: str
    model: str
    compiler_version: str
    created_at: datetime
    human_interventions: int = 0


class CapabilityArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str
    capability_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    contract: CapabilityContract
    application: ApplicationRef
    entry: Entry
    targets: dict[str, NamedTarget]
    steps: tuple[Step, ...] = Field(min_length=1)
    handlers: tuple[Handler, ...] = ()
    success: Predicate
    extractions: tuple[Extraction, ...] = Field(min_length=1)
    provenance: Provenance

    @field_validator("schema_version")
    @classmethod
    def _supported_major(cls, version: str) -> str:
        major, _, _ = version.partition(".")
        if not major.isdigit() or int(major) != SUPPORTED_SCHEMA_MAJOR:
            raise ValueError(f"unsupported schema major version '{version}'")
        return version

    @model_validator(mode="after")
    def _references_are_consistent(self) -> CapabilityArtifact:
        problems = ArtifactReferenceChecker(self).problems()
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def capability_id(self) -> str:
        return self.contract.capability_id

    def content_sha256(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def visual_templates(self) -> set[str]:
        return {
            strategy.template_sha256
            for target in self.targets.values()
            for strategy in target.strategies
            if isinstance(strategy, VisualTarget)
        }


class ArtifactReferenceChecker:
    def __init__(self, artifact: CapabilityArtifact) -> None:
        self._artifact = artifact
        self._inputs = {field.name for field in artifact.contract.inputs}

    def problems(self) -> list[str]:
        return [
            *self._duplicate_steps(),
            *self._unknown_targets(),
            *self._unknown_inputs(),
            *self._unknown_bindings(),
            *self._extraction_mismatch(),
            *self._undeclared_outcomes(),
            *self._undeclared_features(),
            *self._output_rows(),
        ]

    def _output_rows(self) -> list[str]:
        return output_row_problems(
            {
                extraction.output: self._artifact.targets[extraction.target_ref].strategies
                for extraction in self._artifact.extractions
                if extraction.target_ref in self._artifact.targets
            }
        )

    def _duplicate_steps(self) -> list[str]:
        ids = [step.id for step in self._artifact.steps]
        return [f"duplicate step id '{step_id}'" for step_id in sorted({i for i in ids if ids.count(i) > 1})]

    def _referenced_targets(self) -> set[str]:
        artifact = self._artifact
        refs: set[str] = set(target_refs(artifact.success))
        for step in artifact.steps:
            if (ref := step.target_ref()) is not None:
                refs.add(ref)
            refs |= target_refs(step.postcondition)
            if step.precondition is not None:
                refs |= target_refs(step.precondition)
        for handler in artifact.handlers:
            refs |= target_refs(handler.when)
            if isinstance(handler.response, RecoverByClickResponse):
                refs.add(handler.response.target_ref)
        refs |= {extraction.target_ref for extraction in artifact.extractions}
        return refs

    def _unknown_targets(self) -> list[str]:
        return [f"unknown target '{ref}'" for ref in sorted(self._referenced_targets() - set(self._artifact.targets))]

    def _value_refs(self) -> list[InputRef | BindingRef]:
        artifact = self._artifact
        predicates: list[Predicate] = [artifact.success]
        refs: list[object] = []
        for step in artifact.steps:
            predicates.append(step.postcondition)
            if step.precondition is not None:
                predicates.append(step.precondition)
            if isinstance(step.action, FillAction):
                refs.append(step.action.value)
            if isinstance(step.action, NavigateAction):
                refs.append(step.action.base_url)
        predicates.extend(handler.when for handler in artifact.handlers)
        for predicate in predicates:
            refs.extend(value_refs(predicate))
        for target in artifact.targets.values():
            for strategy in target.strategies:
                if isinstance(strategy, WebTarget) and isinstance(strategy.locator, TableCellLocator):
                    refs.append(strategy.locator.row_key)
        return [ref for ref in refs if isinstance(ref, InputRef | BindingRef)]

    def _unknown_inputs(self) -> list[str]:
        names = {ref.name for ref in self._value_refs() if isinstance(ref, InputRef)}
        return [f"unknown input '{name}'" for name in sorted(names - self._inputs)]

    def _unknown_bindings(self) -> list[str]:
        declared = set(self._artifact.entry.required_bindings)
        names = {ref.name for ref in self._value_refs() if isinstance(ref, BindingRef)}
        return [f"undeclared binding '{name}'" for name in sorted(names - declared)]

    def _extraction_mismatch(self) -> list[str]:
        declared = {field.name for field in self._artifact.contract.outputs}
        extracted = [extraction.output for extraction in self._artifact.extractions]
        problems = [f"output '{name}' has no extraction" for name in sorted(declared - set(extracted))]
        problems += [f"extraction for undeclared output '{name}'" for name in sorted(set(extracted) - declared)]
        return problems

    def _undeclared_outcomes(self) -> list[str]:
        declared = set(self._artifact.contract.business_outcomes)
        codes = {
            handler.response.code
            for handler in self._artifact.handlers
            if isinstance(handler.response, BusinessOutcomeResponse)
        }
        return [f"business outcome '{code}' is not declared by the contract" for code in sorted(codes - declared)]

    def _undeclared_features(self) -> list[str]:
        if self._artifact.visual_templates() and "visual.template" not in self._artifact.application.requires:
            return ["visual targets require the 'visual.template' surface feature"]
        return []
