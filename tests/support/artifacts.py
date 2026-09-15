from __future__ import annotations

from datetime import UTC, datetime

from interface_ai_challenge.domain.artifact import (
    SCHEMA_VERSION,
    ApplicationRef,
    CapabilityArtifact,
    Entry,
    Extraction,
    Provenance,
)
from interface_ai_challenge.domain.geometry import Point, Size
from interface_ai_challenge.domain.predicates import AllOf, FieldEquals, TextEquals, Visible
from interface_ai_challenge.domain.steps import ClickAction, Effect, FillAction, NavigateAction, Origin, Step
from interface_ai_challenge.domain.targets import (
    FrameRef,
    LabeledFieldLocator,
    LabeledValueLocator,
    NamedTarget,
    RoleLocator,
    TableCellLocator,
    VisualTarget,
    WebTarget,
)
from interface_ai_challenge.domain.values import BindingRef, InputRef, LiteralValue
from support import configs

WORKSPACE = FrameRef(names=("workspace",))


def _web(locator: object, rationale: str) -> NamedTarget:
    return NamedTarget(strategies=(WebTarget(frame=WORKSPACE, locator=locator),), rationale=rationale)  # type: ignore[arg-type]


def savings_targets(search_button_name: str = "Search") -> dict[str, NamedTarget]:
    savings = LiteralValue(value="Savings")
    return {
        **configs.profile().targets,
        "member_id_field": _web(LabeledFieldLocator(label="Member ID"), "field by label"),
        "search_button": _web(RoleLocator(role="button", name=search_button_name), "button by name"),
        "open_link_in_row": _web(
            TableCellLocator(
                column="Action",
                row_key_column="Member ID",
                row_key=InputRef(name="member_id"),
                control=RoleLocator(role="link", name="Open"),
            ),
            "link in the row of the requested member",
        ),
        "accounts_visual": NamedTarget(
            strategies=(
                VisualTarget(
                    template_sha256="0" * 64,
                    template_size=Size(width=96, height=30),
                    click_offset=Point(x=48, y=15),
                    viewport=Size(width=1280, height=800),
                ),
            ),
            rationale="canvas control",
        ),
        "output_account_type": _web(
            TableCellLocator(column="Account type", row_key_column="Account type", row_key=savings), "type cell"
        ),
        "output_balance": _web(
            TableCellLocator(column="Current balance", row_key_column="Account type", row_key=savings), "balance cell"
        ),
        "output_currency": _web(
            TableCellLocator(column="Currency", row_key_column="Account type", row_key=savings), "currency cell"
        ),
        "identity_member_id": _web(LabeledValueLocator(label="Member ID"), "identity value"),
    }


def savings_artifact(search_button_name: str = "Search", timeout_ms: int = 400) -> CapabilityArtifact:
    member_id = InputRef(name="member_id")
    success = AllOf(
        predicates=(
            TextEquals(target_ref="identity_member_id", value=member_id),
            Visible(target_ref="output_account_type"),
            Visible(target_ref="output_balance"),
            Visible(target_ref="output_currency"),
        )
    )
    steps = (
        Step(
            id="s00_open_entry",
            action=NavigateAction(base_url=BindingRef(name="base_url"), path="/desk"),
            postcondition=Visible(target_ref="member_id_field"),
            effect=Effect.NAVIGATION,
            resume_boundary=True,
            timeout_ms=timeout_ms,
            origin=Origin.COMPILED,
        ),
        Step(
            id="s01_fill_member_id_field",
            action=FillAction(target_ref="member_id_field", value=member_id),
            precondition=Visible(target_ref="member_id_field"),
            postcondition=FieldEquals(target_ref="member_id_field", value=member_id),
            effect=Effect.REVERSIBLE_INPUT,
            timeout_ms=timeout_ms,
            origin=Origin.DISCOVERED,
        ),
        Step(
            id="s02_click_search_button",
            action=ClickAction(target_ref="search_button"),
            precondition=Visible(target_ref="search_button"),
            postcondition=Visible(target_ref="open_link_in_row"),
            effect=Effect.SUBMIT,
            timeout_ms=timeout_ms,
            origin=Origin.DISCOVERED,
        ),
        Step(
            id="s03_click_open_link_in_row",
            action=ClickAction(target_ref="open_link_in_row"),
            precondition=Visible(target_ref="open_link_in_row"),
            postcondition=Visible(target_ref="accounts_visual"),
            effect=Effect.NAVIGATION,
            timeout_ms=timeout_ms,
            origin=Origin.DISCOVERED,
        ),
        Step(
            id="s04_click_accounts_visual",
            action=ClickAction(target_ref="accounts_visual"),
            precondition=Visible(target_ref="accounts_visual"),
            postcondition=success,
            effect=Effect.NAVIGATION,
            timeout_ms=timeout_ms,
            origin=Origin.DISCOVERED,
        ),
    )
    profile = configs.profile()
    outcomes = ("account_not_found", "member_not_found", "validation_rejected")
    return CapabilityArtifact(
        schema_version=SCHEMA_VERSION,
        capability_version="1.0.0",
        contract=configs.contract().model_copy(update={"business_outcomes": outcomes}),
        application=ApplicationRef(
            product=profile.product,
            ui_family=profile.ui_family,
            requires=("visual.template", "web.frames", "web.tables"),
        ),
        entry=Entry(path="/desk"),
        targets=savings_targets(search_button_name),
        steps=steps,
        handlers=profile.handlers,
        success=success,
        extractions=(
            Extraction(output="account_type", target_ref="output_account_type"),
            Extraction(output="balance", target_ref="output_balance"),
            Extraction(output="currency", target_ref="output_currency"),
        ),
        provenance=Provenance(
            discovery_run_id="fixture",
            provider="fixture",
            model="none",
            compiler_version="test",
            created_at=datetime(2026, 9, 14, tzinfo=UTC),
        ),
    )
