from dataclasses import replace

import pytest
from pydantic import ValidationError
from support import configs

from interface_ai_challenge.application.discovery.goal import GoalError, contract_from_goal
from interface_ai_challenge.application.discovery.placeholders import PlaceholderError, PlaceholderResolver
from interface_ai_challenge.domain.contract import (
    CapabilityContract,
    DecimalMoneyParser,
    EnumParser,
    output_row_problems,
)
from interface_ai_challenge.domain.errors import ExtractionError, InputValidationError
from interface_ai_challenge.domain.targets import FrameRef, TableCellLocator, WebTarget
from interface_ai_challenge.domain.values import InputRef, LiteralValue, ValueResolver
from interface_ai_challenge.ports.model import GoalInterpretation, GoalOutput, GoalParameter


@pytest.mark.parametrize(
    ("raw", "parsed"),
    [("$1,250.45", "1250.45"), ("$12,980.07", "12980.07"), ("88.1", "88.10"), (" $0.00 ", "0.00")],
)
def test_money_parser_uses_decimal_strings(raw: str, parsed: str) -> None:
    assert DecimalMoneyParser().parse(raw) == parsed


@pytest.mark.parametrize("raw", ["1.250,45", "USD 10", "", "$1.234"])
def test_money_parser_rejects_ambiguous_text(raw: str) -> None:
    with pytest.raises(ExtractionError):
        DecimalMoneyParser().parse(raw)


def test_enum_parser_maps_visible_labels() -> None:
    parser = EnumParser(mapping={"Savings": "savings"})
    assert parser.parse(" Savings ") == "savings"
    with pytest.raises(ExtractionError):
        parser.parse("Checking")


@pytest.mark.parametrize(
    ("inputs", "field"),
    [
        ({}, "member_id"),
        ({"member_id": ""}, "member_id"),
        ({"member_id": "M-" + "9" * 40}, "member_id"),
        ({"member_id": "M-1", "pin": "1234"}, "pin"),
        ({"member_id": 10023}, "member_id"),
    ],
)
def test_contract_validates_inputs_before_any_action(inputs: dict[str, object], field: str) -> None:
    with pytest.raises(InputValidationError) as error:
        configs.contract().validate_inputs(inputs)
    assert error.value.field == field


def test_placeholders_bind_by_reference() -> None:
    resolver = PlaceholderResolver(ValueResolver({"member_id": "M-10023"}, {}))
    assert resolver.input_ref("{{inputs.member_id}}") == InputRef(name="member_id")
    assert resolver.resolve("{{ inputs.member_id }}") == "M-10023"
    assert resolver.resolve("plain text") == "plain text"
    with pytest.raises(PlaceholderError):
        resolver.resolve("id {{inputs.member_id}}")
    with pytest.raises(PlaceholderError):
        resolver.resolve("{{inputs.password}}")


def test_contract_goal_must_be_parameterized() -> None:
    data = configs.contract().model_dump(mode="json")
    data["goal"] = "Look up the member M-10023 and read the savings balance."
    with pytest.raises(ValidationError, match="placeholder"):
        CapabilityContract.model_validate(data)


def _cell(column: str, row: str) -> WebTarget:
    locator = TableCellLocator(column=column, row_key_column="Account type", row_key=LiteralValue(value=row))
    return WebTarget(frame=FrameRef(names=("workspace",)), locator=locator)


def test_table_outputs_in_one_frame_must_share_a_row() -> None:
    same = {"account_type": [_cell("Account type", "Savings")], "balance": [_cell("Current balance", "Savings")]}
    assert output_row_problems(same) == []
    mixed = {**same, "balance": [_cell("Current balance", "Checking")]}
    assert output_row_problems(mixed) == ["outputs ['account_type', 'balance'] must come from the same table row"]


GOAL = "look up member M-10023 and read their current savings balance"


def _interpretation(**changes: object) -> GoalInterpretation:
    interpretation = GoalInterpretation(
        capability_id="member.savings_balance",
        description="Reads the savings balance of member M-10023.",
        parameters=(GoalParameter("member_id", "Member ID", "M-10023", identifies_record=True),),
        outputs=(GoalOutput("balance", "Current savings balance", "decimal_money"),),
    )
    return replace(interpretation, **changes)  # type: ignore[arg-type]


def test_goal_becomes_a_parameterized_contract_without_raw_values() -> None:
    contract, inputs = contract_from_goal(GOAL, _interpretation())
    assert inputs == {"member_id": "M-10023"}
    assert contract.goal == "look up member {{inputs.member_id}} and read their current savings balance"
    assert contract.inputs[0].identifies_record and contract.outputs[0].sensitive
    assert isinstance(contract.outputs[0].parser, DecimalMoneyParser)
    assert "M-10023" not in contract.model_dump_json()


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"parameters": (GoalParameter("member_id", "Member ID", "M-10024", identifies_record=True),)}, "literally"),
        ({"parameters": ()}, "names no value"),
        ({"outputs": ()}, "outputs"),
        ({"clarification": "Which account type?"}, "Which account type?"),
    ],
)
def test_unusable_goal_interpretations_are_rejected(changes: dict[str, object], problem: str) -> None:
    with pytest.raises(GoalError) as error:
        contract_from_goal(GOAL, _interpretation(**changes))
    assert problem in " ".join(error.value.problems)
