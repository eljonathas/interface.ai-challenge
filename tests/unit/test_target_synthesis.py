from interface_ai_challenge.application.discovery.target_synthesis import WebTargetSynthesizer
from interface_ai_challenge.domain.elements import ElementDescriptor
from interface_ai_challenge.domain.geometry import Rect
from interface_ai_challenge.domain.targets import (
    FrameRef,
    LabeledFieldLocator,
    LabeledValueLocator,
    RoleLocator,
    TableCellLocator,
    WebTarget,
)
from interface_ai_challenge.domain.values import InputRef, LiteralValue, ValueResolver

WORKSPACE = FrameRef(names=("workspace",))
VALUES = ValueResolver({"member_id": "M-10023"}, {})


def _descriptor(**fields: object) -> ElementDescriptor:
    base: dict[str, object] = {
        "frame": WORKSPACE,
        "role": "cell",
        "name": "",
        "rect": Rect(x=0, y=0, width=1, height=1),
    }
    base.update(fields)
    return ElementDescriptor(**base)  # type: ignore[arg-type]


def test_row_link_is_bound_to_the_input_value_not_to_its_position() -> None:
    descriptor = _descriptor(
        role="link",
        name="Open",
        column_header="Action",
        row_cells={"Member ID": "M-10023", "Name": "Alice Carter", "Branch": "Downtown", "Action": "Open"},
    )
    proposals = WebTargetSynthesizer().propose_for_control(descriptor, VALUES)
    first = proposals[0]
    assert first.parametric
    assert first.target == WebTarget(
        frame=WORKSPACE,
        locator=TableCellLocator(
            column="Action",
            row_key_column="Member ID",
            row_key=InputRef(name="member_id"),
            control=RoleLocator(role="link", name="Open"),
        ),
    )
    assert "Alice" not in str([proposal.target for proposal in proposals])


def test_value_cells_use_stable_labels_and_skip_numeric_row_keys() -> None:
    descriptor = _descriptor(
        column_header="Current balance",
        row_cells={
            "Account type": "Savings",
            "Account number": "••••4821",
            "Current balance": "$1,250.45",
            "Currency": "USD",
        },
    )
    targets = [proposal.target for proposal in WebTargetSynthesizer().propose_for_value(descriptor, VALUES)]
    keys = [target.locator.row_key for target in targets if isinstance(target.locator, TableCellLocator)]
    assert keys == [LiteralValue(value="Savings"), LiteralValue(value="USD")]


def test_text_field_uses_visible_label_and_unnamed_frames_are_refused() -> None:
    field = _descriptor(role="textbox", row_label="Member ID")
    proposals = WebTargetSynthesizer().propose_for_control(field, VALUES)
    assert [proposal.target.locator for proposal in proposals] == [LabeledFieldLocator(label="Member ID")]  # type: ignore[union-attr]
    unnamed = _descriptor(role="textbox", row_label="Member ID", frame_is_named=False)
    assert WebTargetSynthesizer().propose_for_control(unnamed, VALUES) == []


def test_key_value_cell_uses_row_label() -> None:
    proposals = WebTargetSynthesizer().propose_for_value(_descriptor(row_label="Member ID"), VALUES)
    assert [proposal.target.locator for proposal in proposals] == [LabeledValueLocator(label="Member ID")]  # type: ignore[union-attr]
