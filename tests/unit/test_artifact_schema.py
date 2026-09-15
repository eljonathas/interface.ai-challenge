import json
from typing import Any

import pytest
from pydantic import ValidationError
from support.artifacts import savings_artifact

from interface_ai_challenge.adapters.file_store import CapabilityFileRepository
from interface_ai_challenge.domain.artifact import CapabilityArtifact
from interface_ai_challenge.domain.errors import ArtifactIncompatibleError


def _document() -> dict[str, Any]:
    return savings_artifact().model_dump(mode="json")


def test_artifact_round_trips_with_stable_hash() -> None:
    artifact = savings_artifact()
    reloaded = CapabilityArtifact.model_validate_json(json.dumps(artifact.model_dump(mode="json")))
    assert reloaded == artifact
    assert reloaded.content_sha256() == artifact.content_sha256()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(schema_version="1.1"), "unsupported schema major version"),
        (lambda d: d["steps"][2]["action"].update(target_ref="missing_button"), "unknown target 'missing_button'"),
        (lambda d: d["steps"][1]["action"]["value"].update(name="account_number"), "unknown input 'account_number'"),
        (lambda d: d["steps"][0]["action"]["base_url"].update(name="tenant_url"), "undeclared binding 'tenant_url'"),
        (lambda d: d["extractions"].pop(), "output 'currency' has no extraction"),
        (lambda d: d["steps"][1].update(id="s00_open_entry"), "duplicate step id"),
        (lambda d: d["application"].update(requires=["web.frames"]), "visual.template"),
        (lambda d: d["steps"][2]["action"].update(kind="execute_js"), "execute_js"),
    ],
)
def test_invalid_artifacts_are_rejected(mutate: Any, message: str) -> None:
    document = _document()
    mutate(document)
    with pytest.raises(ValidationError) as error:
        CapabilityArtifact.model_validate(document)
    assert message in str(error.value) or "kind" in str(error.value)


def test_undeclared_business_outcome_is_rejected() -> None:
    document = _document()
    document["contract"]["business_outcomes"] = ["member_not_found"]
    with pytest.raises(ValidationError, match="account_not_found"):
        CapabilityArtifact.model_validate(document)


@pytest.mark.parametrize("fallback", [False, True])
def test_artifact_rejects_outputs_from_different_rows(fallback: bool) -> None:
    document = _document()
    strategies = document["targets"]["output_balance"]["strategies"]
    wrong = json.loads(json.dumps(strategies[0]))
    wrong["locator"]["row_key"]["value"] = "Checking"
    if fallback:
        strategies.append(wrong)
    else:
        strategies[0] = wrong
    with pytest.raises(ValidationError, match="same table row"):
        CapabilityArtifact.model_validate(document)


def test_repository_refuses_missing_or_tampered_template(tmp_path: Any) -> None:
    repository = CapabilityFileRepository()
    path = repository.save(savings_artifact(), tmp_path)
    with pytest.raises(ArtifactIncompatibleError, match="missing"):
        repository.load(path)
    template = tmp_path / "assets" / f"{'0' * 64}.png"
    template.parent.mkdir()
    template.write_bytes(b"tampered")
    with pytest.raises(ArtifactIncompatibleError, match="does not match"):
        repository.load(path)
