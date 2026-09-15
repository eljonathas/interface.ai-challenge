from __future__ import annotations

from pydantic import ValidationError

from interface_ai_challenge.application.discovery.placeholders import PlaceholderResolver
from interface_ai_challenge.domain.contract import (
    CapabilityContract,
    DecimalMoneyParser,
    InputField,
    OutputField,
    TextParser,
)
from interface_ai_challenge.domain.errors import InputValidationError
from interface_ai_challenge.ports.model import GoalInterpretation


class GoalError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


def contract_from_goal(goal: str, interpretation: GoalInterpretation) -> tuple[CapabilityContract, dict[str, str]]:
    """Checks the model's reading of a free-text goal and returns a parameterized contract plus discovery inputs.

    Parameter values must be literal parts of the goal. They are replaced by placeholders in every persisted text,
    so the raw goal is never stored. Outputs are treated as sensitive: redaction does not depend on the model.
    """

    if interpretation.clarification:
        raise GoalError([interpretation.clarification])
    parameters = sorted(interpretation.parameters, key=lambda parameter: len(parameter.value), reverse=True)
    problems = [
        f"parameter '{parameter.name}' is not written literally in the goal"
        for parameter in parameters
        if not parameter.value.strip() or parameter.value not in goal
    ]
    names = [parameter.name for parameter in parameters]
    if len(set(names)) != len(names):
        problems.append("parameter names must be unique")
    if not parameters:
        problems.append("the goal names no value that changes per invocation")
    if problems:
        raise GoalError(problems)

    def parameterized(text: str) -> str:
        for parameter in parameters:
            text = text.replace(parameter.value, PlaceholderResolver.placeholder_for(parameter.name))
        return text

    inputs = {parameter.name: parameter.value for parameter in parameters}
    try:
        contract = CapabilityContract(
            capability_id=interpretation.capability_id,
            description=parameterized(interpretation.description),
            goal=parameterized(goal),
            inputs=tuple(
                InputField(
                    name=parameter.name,
                    description=parameterized(parameter.description),
                    identifies_record=parameter.identifies_record,
                )
                for parameter in interpretation.parameters
            ),
            outputs=tuple(
                OutputField(
                    name=output.name,
                    description=parameterized(output.description),
                    parser=DecimalMoneyParser() if output.kind == "decimal_money" else TextParser(),
                    sensitive=True,
                )
                for output in interpretation.outputs
            ),
        )
        contract.validate_inputs(dict(inputs))
    except ValidationError as error:
        # Pydantic messages can echo input values; keep only locations and reasons.
        raise GoalError([f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()]) from None
    except InputValidationError as error:
        raise GoalError([f"parameter '{error.field}' is invalid"]) from None
    if any(value in contract.capability_id for value in inputs.values()):
        raise GoalError(["capability_id must not contain parameter values"])
    return contract, inputs
