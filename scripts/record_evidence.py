"""Records the /evidence folder: real LLM discoveries plus model-free replays of the discovered capability.

Usage:
    uv run --env-file .env python scripts/record_evidence.py

Requires the provider API key. The handoff replay waits for a person to operate the printed console link.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from interface_ai_challenge.adapters.file_store import CapabilityFileRepository, load_model
from interface_ai_challenge.agents import PROVIDER_CREDENTIALS, create_agent, create_goal_interpreter
from interface_ai_challenge.application.discovery.goal import contract_from_goal
from interface_ai_challenge.bootstrap import run_discovery, run_replay
from interface_ai_challenge.demo_app.data import SYNTHETIC_MEMBERS
from interface_ai_challenge.demo_app.scenarios import Scenario
from interface_ai_challenge.demo_app.server import serve_in_thread
from interface_ai_challenge.domain.contract import CapabilityContract
from interface_ai_challenge.domain.policy import PolicyConfig
from interface_ai_challenge.domain.profile import TargetProfile
from interface_ai_challenge.domain.results import RunResult
from interface_ai_challenge.settings import ModelSettings, SystemConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
EVIDENCE = ROOT / "evidence"
CAPABILITY_DIR = EVIDENCE / "capability"
SYSTEM = load_model(CONFIGS / "system.json", SystemConfig)
OPERATOR_PORT = 8765
# Free text, as a caller would write it. Only its parameterized form may reach /evidence.
GOAL = "Look up member M-10023 and read the current balance of their savings account"
CANARIES = {"M-99999", "***-**-"}
for member in SYNTHETIC_MEMBERS.values():
    CANARIES.update((member.id, member.name, member.ssn, member.birth_date))
    for account in member.accounts:
        CANARIES.update((account.balance.removeprefix("$"), account.balance.removeprefix("$").replace(",", "")))


@dataclass(frozen=True)
class ReplayCase:
    folder: str
    scenario: Scenario
    inputs_file: str
    purpose: str
    human_operator: bool = False
    expected: tuple[str, str | None] = ("success", None)


REPLAY_CASES = (
    ReplayCase("replay-success-member-b", Scenario.NORMAL, "member-b.json", "different parameter than discovery"),
    ReplayCase(
        "replay-outcome-member-not-found",
        Scenario.NORMAL,
        "member-missing.json",
        "business outcome",
        expected=("business_outcome", "member_not_found"),
    ),
    ReplayCase(
        "replay-outcome-account-not-found",
        Scenario.NORMAL,
        "member-no-savings.json",
        "business outcome",
        expected=("business_outcome", "account_not_found"),
    ),
    ReplayCase(
        "replay-outcome-validation-rejected",
        Scenario.NORMAL,
        "member-invalid-format.json",
        "business outcome",
        expected=("business_outcome", "validation_rejected"),
    ),
    ReplayCase("replay-recovery-maintenance-notice", Scenario.MAINTENANCE, "member-b.json", "recoverable interstitial"),
    ReplayCase("replay-recovery-transient-error", Scenario.TRANSIENT_ERROR, "member-b.json", "recoverable read error"),
    ReplayCase("replay-recovery-slow-load", Scenario.SLOW, "member-b.json", "slow response within step deadline"),
    ReplayCase("replay-visual-canvas-shifted", Scenario.CANVAS_SHIFT, "member-b.json", "visual anchor relocated"),
    ReplayCase(
        "replay-failure-permission-denied",
        Scenario.PERMISSION_DENIED,
        "member-b.json",
        "hard failure",
        expected=("failure", "permission_denied"),
    ),
    ReplayCase(
        "replay-failure-ambiguous-canvas",
        Scenario.DUPLICATE_CANVAS_CONTROL,
        "member-b.json",
        "ambiguity",
        expected=("failure", "target_ambiguous"),
    ),
    ReplayCase(
        "replay-escalation-no-operator",
        Scenario.UNKNOWN_DIALOG,
        "member-b.json",
        "stuck, no operator",
        expected=("failure", "escalation_unavailable"),
    ),
    ReplayCase(
        "replay-handoff-human-operator",
        Scenario.SESSION_EXPIRED,
        "member-b.json",
        "session expiry handed to a person in the operator console",
        human_operator=True,
    ),
)


def load_configs(base_url: str) -> tuple[PolicyConfig, TargetProfile]:
    policy = load_model(CONFIGS / "policy.json", PolicyConfig).model_copy(update={"allowed_origins": (base_url,)})
    return policy, load_model(CONFIGS / "target-profile.json", TargetProfile)


def inputs(name: str) -> dict[str, object]:
    """The input name is chosen during discovery, so bind the synthetic value to the artifact's single input."""
    contract = CapabilityFileRepository().load(CAPABILITY_DIR / "capability.json").contract
    if len(contract.inputs) != 1:
        raise SystemExit(f"expected one input, got {[field.name for field in contract.inputs]}")
    (value,) = json.loads((CONFIGS / "inputs" / name).read_text()).values()
    return {contract.inputs[0].name: value}


async def discover(
    settings: ModelSettings,
) -> tuple[dict[str, object], CapabilityContract, dict[str, str]]:
    with serve_in_thread(Scenario.NORMAL) as base_url:
        policy, profile = load_configs(base_url)
        contract, discovery_inputs = contract_from_goal(GOAL, await create_goal_interpreter(settings).interpret(GOAL))
        report = await run_discovery(
            agent=create_agent(settings),
            contract=contract,
            inputs=discovery_inputs,
            entry_url=f"{base_url}/desk",
            output_dir=CAPABILITY_DIR,
            policy=policy,
            profile=profile,
            runtime=SYSTEM.runtime(),
        )
    shutil.move(str(CAPABILITY_DIR / "discovery-run"), str(EVIDENCE / "discovery-live"))
    if not report.completed:
        raise SystemExit(f"discovery did not complete: {report.reason} {report.problems}")
    summary: dict[str, object] = {
        "provider": settings.provider,
        "model": settings.model,
        "goal_template": contract.goal,
        "turns": report.turns,
        "usage": report.usage.__dict__,
    }
    return summary, contract, discovery_inputs


async def injection_discovery(
    settings: ModelSettings, contract: CapabilityContract, discovery_inputs: dict[str, str]
) -> dict[str, object]:
    """A second real discovery on a page that tells agents to leave the application. Nothing is scripted."""
    folder = EVIDENCE / "discovery-prompt-injection"
    with tempfile.TemporaryDirectory() as scratch, serve_in_thread(Scenario.INJECTION) as base_url:
        policy, profile = load_configs(base_url)
        report = await run_discovery(
            agent=create_agent(settings),
            contract=contract,
            inputs=discovery_inputs,
            entry_url=f"{base_url}/desk",
            output_dir=Path(scratch),
            policy=policy,
            profile=profile,
            runtime=SYSTEM.runtime(),
        )
        shutil.move(str(Path(scratch) / "discovery-run"), str(folder))
    events = [json.loads(line)["event"] for line in (folder / "events.jsonl").read_text().splitlines()]
    return {
        "completed": report.completed,
        "reason": report.reason,
        "turns": report.turns,
        "blocked_requests": events.count("request_blocked"),
        "blocked_actions": events.count("agent_action_blocked"),
    }


async def replay(case: ReplayCase) -> RunResult:
    with serve_in_thread(case.scenario) as base_url:
        policy, profile = load_configs(base_url)
        if case.human_operator:
            print(f"\n{case.folder}: waiting for a person. Open the console link printed below, take control,")
            print("sign in with demo/demo and hand control back.")
        return await run_replay(
            artifact_path=CAPABILITY_DIR / "capability.json",
            inputs=inputs(case.inputs_file),
            bindings={"base_url": base_url},
            run_dir=EVIDENCE / case.folder,
            policy=policy,
            redaction=profile.redaction,
            runtime=SYSTEM.runtime(OPERATOR_PORT if case.human_operator else None),
        )


def assert_no_canaries() -> None:
    leaks = [
        f"{path.relative_to(ROOT)}: {canary}"
        for path in EVIDENCE.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}
        for canary in CANARIES
        if canary in path.read_text(encoding="utf-8")
    ]
    if leaks:
        raise SystemExit("sensitive canaries found in evidence:\n" + "\n".join(leaks))


def versions() -> dict[str, str]:
    packages = ("playwright", "openai", "pydantic", "fastapi", "opencv-python-headless")
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in packages}}


def write_readme(manifest: dict[str, object]) -> None:
    discovery = manifest["discovery"]
    injection = manifest["prompt_injection_discovery"]
    replays = manifest["replays"]
    assert isinstance(discovery, dict) and isinstance(injection, dict) and isinstance(replays, list)
    completed = "yes" if injection["completed"] else "no"
    artifact = json.loads((CAPABILITY_DIR / "capability.json").read_text())
    rows = "\n".join(
        f"| [{run['folder']}]({run['folder']}/result.json) | {run['purpose']} | "
        f"`{run['status']}`{': `' + run['code'] + '`' if run['code'] else ''}"
        f"{' (assisted)' if run['assisted'] else ''} |"
        for run in replays
    )
    (EVIDENCE / "README.md").write_text(
        f"""# Evidence

Generated at {manifest['generated_at']} against the local demo application with synthetic data.
[manifest.json](manifest.json) records versions, token usage, the capability hash and all replay results.

## Real discovery and deterministic replay

- Discovery started from a free-text goal with a concrete member ID. The model proposed the contract and the code kept
  only its parameterized form: `{discovery.get('goal_template')}`.
- [discovery-live/events.jsonl](discovery-live/events.jsonl): a real **{discovery['provider']} / {discovery['model']}**
  run, completed in **{discovery['turns']} turns**. The model chose its actions from live screenshots.
- [capability/capability.json](capability/capability.json): **{len(artifact['steps'])} steps**, typed parameters,
  outputs, checkpoints, targets and authored exception handlers; an image template is stored in `capability/assets/`.
- Every replay below uses this same artifact. Successful replays use a different member from discovery.
  The script checks expected status/code, returned outputs, assisted status and screenshots for failures.
- Replay never instantiates a model client. Each `replay_started` event records `llm_calls: 0`; the unit suite
  separately checks that the replay path does not import the model SDK.

| Result file | Purpose | Verified result |
| --- | --- | --- |
{rows}

## Reading the evidence

`events.jsonl` contains ordered events with timestamps and run IDs. Discovery records model request IDs, token usage,
structured actions and target rationale. `observation_refresh` marks the screenshot exchange after function-only
replies: automation executes no stale actions while the model obtains the current observation.

`result.json` reports success, a business outcome or a failure. Failure results identify the step, the expected
condition and what was observed, and include a masked screenshot reference. The caller receives actual output values;
persisted balances are redacted.

`screenshots/` contains masked evidence. Member identifiers, names, SSNs, birth dates and balance cells are covered by
the application profile. Free model text is omitted from logs because it can repeat data from the screen. The export
scans text files for synthetic member identifiers, names, birth dates, SSNs and balances and fails on a match.

## Human and adversarial runs

- The model proposes inputs/outputs from the goal (checked in code) and discovers navigation, extraction and identity
  targets. Known error handlers, masking regions and permissions are authored application configuration.
- [replay-handoff-human-operator](replay-handoff-human-operator/events.jsonl): a person took control of the live
  session in the operator console, signed in again and handed control back. Events include `control_claimed`,
  `human_action`, `control_returned` and `resume_reconciled`.
- [discovery-prompt-injection/events.jsonl](discovery-prompt-injection/events.jsonl): a second real discovery on a page
  instructing agents to open an external site. Completed: **{completed}**; blocked browser requests:
  **{injection['blocked_requests']}**; blocked agent actions: **{injection['blocked_actions']}**. Zero blocked requests
  means the model did not follow the injected instruction; the allowlist is the enforcement either way.

## Reproduce

```bash
uv run --env-file .env python scripts/record_evidence.py   # paid discoveries, replays and a manual handoff
```
""",
        encoding="utf-8",
    )


async def main(model: str | None) -> None:
    settings = SYSTEM.llm_settings(model=model)
    credential = PROVIDER_CREDENTIALS.get(settings.provider)
    if credential and not os.environ.get(credential):
        raise SystemExit(f"set {credential}: evidence is only produced from real model runs")
    for folder in EVIDENCE.glob("*"):
        if folder.is_dir():
            shutil.rmtree(folder)
    discovery, contract, discovery_inputs = await discover(settings)
    results = []
    for case in REPLAY_CASES:
        result = await replay(case)
        actual = (result.status, getattr(result, "code", None))
        if actual != case.expected or result.assisted != case.human_operator:
            raise SystemExit(f"{case.folder}: expected {case.expected}, got {actual}, assisted={result.assisted}")
        if result.status == "success" and "12980.07" not in result.outputs.values():
            raise SystemExit(f"{case.folder}: returned outputs do not match the requested synthetic member")
        if result.status == "failure" and (
            not result.evidence_ref or not (EVIDENCE / case.folder / result.evidence_ref).is_file()
        ):
            raise SystemExit(f"{case.folder}: failure screenshot is missing")
        summary = {
            "folder": case.folder,
            "scenario": case.scenario.value,
            "inputs": case.inputs_file,
            "purpose": case.purpose,
            "status": result.status,
            "code": getattr(result, "code", None),
            "assisted": result.assisted,
        }
        results.append(summary)
        print(json.dumps(summary))
    injection = await injection_discovery(settings, contract, discovery_inputs)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "versions": versions(),
        "capability_sha256": (CAPABILITY_DIR / "capability.sha256").read_text().strip(),
        "discovery": discovery,
        "replays": results,
        "prompt_injection_discovery": injection,
        "reproduce": "uv run --env-file .env python scripts/record_evidence.py",
        "synthetic_data_only": True,
    }
    (EVIDENCE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_readme(manifest)
    assert_no_canaries()
    print(json.dumps({"discovery": discovery, "prompt_injection_discovery": injection}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    asyncio.run(main(parser.parse_args().model))
