from __future__ import annotations

from pathlib import Path

from interface_ai_challenge.adapters.file_store import load_model
from interface_ai_challenge.domain.contract import CapabilityContract
from interface_ai_challenge.domain.policy import PolicyConfig
from interface_ai_challenge.domain.profile import TargetProfile

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def policy(origin: str = "http://127.0.0.1:8000") -> PolicyConfig:
    return load_model(CONFIGS / "policy.json", PolicyConfig).model_copy(update={"allowed_origins": (origin,)})


def profile() -> TargetProfile:
    return load_model(CONFIGS / "target-profile.json", TargetProfile)


def contract() -> CapabilityContract:
    # Fixed contract for the scripted model double; real discovery derives the contract from --goal.
    return load_model(Path(__file__).resolve().parents[1] / "fixtures" / "savings-contract.json", CapabilityContract)
