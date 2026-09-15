from __future__ import annotations

from collections.abc import Sequence

from interface_ai_challenge.application.target_resolution import NamedTargetResolver
from interface_ai_challenge.domain.artifact import Extraction
from interface_ai_challenge.domain.contract import CapabilityContract
from interface_ai_challenge.domain.errors import ExtractionError, SurfaceError
from interface_ai_challenge.domain.values import ValueResolver
from interface_ai_challenge.ports.surface import ElementActions


class OutputExtractor:
    def __init__(self, resolver: NamedTargetResolver, actions: ElementActions, values: ValueResolver) -> None:
        self._resolver = resolver
        self._actions = actions
        self._values = values

    async def extract(self, contract: CapabilityContract, extractions: Sequence[Extraction]) -> dict[str, str]:
        outputs: dict[str, str] = {}
        for extraction in extractions:
            candidate = (await self._resolver.resolve(extraction.target_ref, self._values)).unique
            if candidate is None:
                raise ExtractionError(f"output '{extraction.output}' is not uniquely visible")
            try:
                raw = await self._actions.read_text(candidate)
            except SurfaceError as error:
                raise ExtractionError(f"output '{extraction.output}' could not be read") from error
            try:
                outputs[extraction.output] = contract.output_named(extraction.output).parser.parse(raw)
            except ExtractionError as error:
                raise ExtractionError(f"output '{extraction.output}': {error}") from error
        return outputs
