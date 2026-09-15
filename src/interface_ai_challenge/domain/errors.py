from __future__ import annotations


class DomainError(Exception):
    pass


class InputValidationError(DomainError):
    def __init__(self, field: str, problem: str) -> None:
        super().__init__(f"input '{field}' {problem}")
        self.field = field
        self.problem = problem


class ExtractionError(DomainError):
    pass


class ArtifactIncompatibleError(DomainError):
    pass


class PolicyViolationError(DomainError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ControlError(DomainError):
    pass


class SurfaceError(DomainError):
    pass


class StaleObservationError(DomainError):
    pass


class CompilationError(DomainError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems
