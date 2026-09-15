from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Annotated

import typer

from interface_ai_challenge.adapters.file_store import load_json_object, load_model
from interface_ai_challenge.domain.policy import PolicyConfig
from interface_ai_challenge.domain.profile import TargetProfile
from interface_ai_challenge.settings import SystemConfig

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Computer-use discovery and deterministic replay.")

_EXIT_CODES = {"success": 0, "business_outcome": 2, "failure": 1}

PolicyOption = Annotated[Path, typer.Option(help="Allowlist and effect policy JSON.")]
ProfileOption = Annotated[Path, typer.Option(help="Authored target profile JSON (handlers, redaction).")]
ConfigOption = Annotated[
    Path, typer.Option(help="System settings JSON: model, browser (headless), operator timeout, console logs.")
]
OperatorPortOption = Annotated[
    int | None, typer.Option(help="Serve the operator console on this loopback port to enable human handoff.")
]


@app.command("demo-app")
def demo_app(
    scenario: Annotated[str, typer.Option(help="Scenario injected into the stand-in application.")] = "normal",
    port: int = 8000,
) -> None:
    """Run the Legacy Credit Union Desk stand-in application."""
    import uvicorn

    from interface_ai_challenge.demo_app.app import create_app
    from interface_ai_challenge.demo_app.scenarios import Scenario

    uvicorn.run(create_app(Scenario(scenario)), host="127.0.0.1", port=port, log_level="warning")


@app.command()
def discover(
    target: Annotated[str, typer.Option(help="Entry URL of the application.")],
    goal: Annotated[
        str,
        typer.Option(help="Goal with concrete values, e.g. 'look up member M-10023 and read their savings balance'."),
    ],
    output: Annotated[
        Path, typer.Option(help="Directory for the capability artifact and discovery evidence.")
    ] = Path("runs/capability"),
    policy: PolicyOption = Path("configs/policy.json"),
    profile: ProfileOption = Path("configs/target-profile.json"),
    provider: Annotated[str | None, typer.Option(help="Model provider adapter (overrides the config).")] = None,
    model: Annotated[str | None, typer.Option(help="Model id (overrides the config).")] = None,
    operator_port: OperatorPortOption = None,
    config: ConfigOption = Path("configs/system.json"),
) -> None:
    """Let the model accomplish the goal once, then compile the verified run into a capability artifact."""
    from interface_ai_challenge.agents import PROVIDER_CREDENTIALS, create_agent, create_goal_interpreter
    from interface_ai_challenge.application.discovery.goal import GoalError, contract_from_goal
    from interface_ai_challenge.bootstrap import run_discovery

    system = load_model(config, SystemConfig)
    model_settings = system.llm_settings(provider=provider, model=model)
    credential = PROVIDER_CREDENTIALS.get(model_settings.provider)
    if credential and not os.environ.get(credential):
        raise typer.BadParameter(
            f"set {credential} in the environment to run discovery with '{model_settings.provider}'"
        )
    interpretation = asyncio.run(create_goal_interpreter(model_settings).interpret(goal))
    try:
        task_contract, task_inputs = contract_from_goal(goal, interpretation)
    except GoalError as error:
        typer.echo(json.dumps({"completed": False, "reason": "goal not usable", "problems": error.problems}))
        raise typer.Exit(1) from None
    report = asyncio.run(
        run_discovery(
            agent=create_agent(model_settings),
            contract=task_contract,
            inputs=task_inputs,
            entry_url=target,
            output_dir=output,
            policy=load_model(policy, PolicyConfig),
            profile=load_model(profile, TargetProfile),
            runtime=system.runtime(operator_port),
        )
    )
    typer.echo(
        json.dumps(
            {
                "capability_id": task_contract.capability_id,
                "inputs": [field.name for field in task_contract.inputs],
                "completed": report.completed,
                "reason": report.reason,
                "artifact": str(report.artifact_path) if report.artifact_path else None,
                "evidence": str(report.run_dir),
                "problems": list(report.problems),
                "turns": report.turns,
                "usage": report.usage.__dict__,
                "provider": model_settings.provider,
                "model": model_settings.model,
            },
            indent=2,
        )
    )
    raise typer.Exit(0 if report.completed else 1)


@app.command()
def replay(
    artifact: Annotated[Path, typer.Option(help="Capability artifact produced by discovery.")],
    inputs: Annotated[Path, typer.Option(help="Inputs JSON for this invocation.")],
    binding: Annotated[Path, typer.Option(help="Tenant/environment bindings JSON, e.g. base_url.")],
    run_dir: Annotated[Path, typer.Option(help="Directory for replay evidence.")],
    policy: PolicyOption = Path("configs/policy.json"),
    profile: ProfileOption = Path("configs/target-profile.json"),
    operator_port: OperatorPortOption = None,
    config: ConfigOption = Path("configs/system.json"),
) -> None:
    from interface_ai_challenge.bootstrap import run_replay

    bindings = {key: str(value) for key, value in load_json_object(binding).items()}
    result = asyncio.run(
        run_replay(
            artifact_path=artifact,
            inputs=load_json_object(inputs),
            bindings=bindings,
            run_dir=run_dir,
            policy=load_model(policy, PolicyConfig),
            redaction=load_model(profile, TargetProfile).redaction,
            runtime=load_model(config, SystemConfig).runtime(operator_port),
        )
    )
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2))
    raise typer.Exit(_EXIT_CODES[result.status])
