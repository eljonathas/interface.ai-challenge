from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from interface_ai_challenge.domain.artifact import CapabilityArtifact
from interface_ai_challenge.domain.errors import ArtifactIncompatibleError

ARTIFACT_FILE = "capability.json"
ASSETS_DIRECTORY = "assets"


class FileAssetStore:
    """Content-addressed binary assets; a read whose hash does not match is refused."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def put(self, content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{digest}.png"
        if not path.exists():
            path.write_bytes(content)
        return digest

    def get(self, sha256: str) -> bytes:
        path = self._directory / f"{sha256}.png"
        if not path.exists():
            raise ArtifactIncompatibleError(f"asset {sha256} is missing")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != sha256:
            raise ArtifactIncompatibleError(f"asset {sha256} does not match its hash")
        return content


class CapabilityFileRepository:
    def save(self, artifact: CapabilityArtifact, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ARTIFACT_FILE
        path.write_text(json.dumps(artifact.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
        (directory / "capability.sha256").write_text(artifact.content_sha256() + "\n", encoding="utf-8")
        return path

    def load(self, path: Path) -> CapabilityArtifact:
        try:
            artifact = CapabilityArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ArtifactIncompatibleError(f"cannot read capability artifact: {error.strerror}") from None
        except ValidationError as error:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()[:5])
            raise ArtifactIncompatibleError(f"invalid capability artifact: {problems}") from None
        assets = self.assets_for(path)
        for sha256 in artifact.visual_templates():
            assets.get(sha256)
        return artifact

    @staticmethod
    def assets_for(artifact_path: Path) -> FileAssetStore:
        return FileAssetStore(artifact_path.parent / ASSETS_DIRECTORY)


def load_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def load_json_object(path: Path) -> dict[str, object]:
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return content
