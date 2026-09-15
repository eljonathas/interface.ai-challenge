# Computer-Use Automation System

An LLM discovers how to do a task once in a legacy back-office UI. The run is compiled into a typed, versioned
**capability artifact**. The artifact is then **replayed deterministically without any model**, with explicit
business outcomes, recoverable conditions, hard failures and human handoff on the same live session.

- Design write-up: [`REPORT.md`](REPORT.md)
- Evidence of a real OpenAI discovery and 12 replays: [`evidence/`](evidence/README.md)

## Layout

```text
src/interface_ai_challenge/
  domain/        pure models: artifact schema, targets, predicates, handlers, results, policy, control states
  ports/         interfaces the application depends on: model, surface, evidence, intervention, imaging
  application/   use cases: discovery loop, recorder, compiler, replay engine, policy guard, handoff
  adapters/      OpenAI computer use, Playwright surface, OpenCV templates, JSONL evidence, operator console
  demo_app/      "Legacy Credit Union Desk", the synthetic stand-in target application
  agents.py      provider registry (one entry per model adapter)
  bootstrap.py   composition root
  cli.py         `cua` command line
configs/         system settings, policy, target profile, bindings, synthetic replay inputs
scripts/         record_evidence.py
tests/unit       fast tests with in-memory fakes (no browser, no model)
tests/integration  end-to-end tests against the demo app in Chromium (scripted model double)
```

## Setup

```bash
uv sync --extra discovery
uv run playwright install chromium
sudo uv run playwright install-deps chromium   # Linux/WSL: system libraries Chromium needs
```

`--extra discovery` installs the OpenAI SDK. Replay does not need it.

## Model configuration

Put `OPENAI_API_KEY` in `.env` (git-ignored) and pass it with `uv run --env-file .env`. Everything else lives in
[`configs/system.json`](configs/system.json), read by `cua discover`, `cua replay` (`--config` to use another file)
and the evidence script:

| Setting | Meaning |
| --- | --- |
| `llm.provider`, `llm.model` | Adapter registered in `agents.py` and model id |
| `llm.reasoning_effort` | `low`, `medium`, ... or `null` |
| `llm.reasoning_summary` | `auto`, `concise`, `detailed` or `null`: asks the provider for reasoning summaries |
| `browser.headless` | `false` opens a visible Chromium window |
| `browser.viewport` | Window size used for screenshots and coordinates |
| `operator.intervention_timeout_seconds` | How long a paused run waits for a person |
| `console.events` | Print every evidence event live to stderr (already redacted, same as `events.jsonl`) |
| `console.model_decisions` | Print each model turn live: reasoning summary, message, actions and signals |

Model text is shown on the terminal only and never written to evidence, because it can repeat screen data; do not
redirect stderr to shared logs when `console.model_decisions` is on. `--provider` and `--model` override the file. Adding another provider (Anthropic, a local
model) means implementing `ports.model.ComputerUseAgent` and registering a factory in `agents.py`.

## What discovery learns

A task is only `--goal` (free text with its concrete values) and `--target` (entry URL). One model call proposes the
contract: parameters, outputs and which parameter identifies the record. Code checks that each parameter value is
written literally in the goal and replaces it with `{{inputs.<name>}}` everywhere it is persisted, so the raw goal is
never stored. Outputs are always treated as sensitive. If the goal is ambiguous, discovery stops with a question.

The model then chooses the navigation from live screenshots. The recorder derives and verifies the navigation targets,
extraction locations and the on-screen value that proves the right record is open (`report_identity`; a text field
or a row selected by the input itself is rejected). Table outputs in one frame must come from the same row.

Application configuration is reused across goals and has defaults: `configs/policy.json` (permissions) and
`configs/target-profile.json` (known error handlers, business outcome codes, masking regions). They are authored, not
learned: one successful run cannot reveal unseen exceptions.

## Demo path

```bash
# Terminal 1: target application. Scenarios: normal, maintenance, slow, transient_error, session_expired,
# permission_denied, unknown_dialog, canvas_shift, duplicate_canvas_control, injection
uv run cua demo-app --scenario normal

# Terminal 2: real LLM discovery (requires a paid API call), then compile the capability
uv run --env-file .env cua discover \
  --goal "Look up member M-10023 and read the current balance of their savings account" \
  --target http://127.0.0.1:8000/desk \
  --output runs/capability

# Deterministic replay with a different member (no model involved). Input names come from `contract.inputs` in the
# artifact; the files in configs/inputs use `member_id`, the name chosen in the committed evidence.
uv run cua replay \
  --artifact runs/capability/capability.json \
  --inputs configs/inputs/member-b.json \
  --binding configs/bindings/local.json \
  --run-dir runs/replay-member-b

# Business outcome
uv run cua replay --artifact runs/capability/capability.json --inputs configs/inputs/member-missing.json \
  --binding configs/bindings/local.json --run-dir runs/replay-missing

# Human handoff: restart the demo app with --scenario session_expired, then
uv run cua replay --artifact runs/capability/capability.json --inputs configs/inputs/member-b.json \
  --binding configs/bindings/local.json --run-dir runs/replay-handoff --operator-port 8765
# open the printed console link, take control, sign in with demo/demo, hand control back
```

Without an API key, replay the artifact already committed in `evidence/capability/capability.json` the same way.

Exit codes of `replay`: `0` success, `2` business outcome, `1` failure. The JSON result is printed to stdout;
evidence (`events.jsonl`, `result.json`, masked screenshots) is written to `--run-dir`.

```bash
uv run --env-file .env python scripts/record_evidence.py          # regenerate /evidence (see below)
```

Evidence is only produced from real model runs: the script refuses to start without the API key, runs the paid
discovery and a second real discovery on the prompt-injection page, and pauses the handoff replay until a person
operates the printed console link (sign in with demo/demo, hand control back).

## Tests

```bash
uv run pytest tests/unit          # no browser, no network
uv run pytest tests/integration   # Chromium; skipped with a hint when it cannot start
uv run ruff check src tests scripts && uv run mypy src
```

The integration discovery is driven by a scripted test double of the model port, never presented as a real LLM run.
