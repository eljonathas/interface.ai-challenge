# Evidence

Generated at 2026-09-15T13:17:26.910519+00:00 against the local demo application with synthetic data.
[manifest.json](manifest.json) records versions, token usage, the capability hash and all replay results.

## Real discovery and deterministic replay

- Discovery started from a free-text goal with a concrete member ID. The model proposed the contract and the code kept
  only its parameterized form: `Look up member {{inputs.member_id}} and read the current balance of their savings account`.
- [discovery-live/events.jsonl](discovery-live/events.jsonl): a real **openai / gpt-5.6-sol**
  run, completed in **7 turns**. The model chose its actions from live screenshots.
- [capability/capability.json](capability/capability.json): **5 steps**, typed parameters,
  outputs, checkpoints, targets and authored exception handlers; an image template is stored in `capability/assets/`.
- Every replay below uses this same artifact. Successful replays use a different member from discovery.
  The script checks expected status/code, returned outputs, assisted status and screenshots for failures.
- Replay never instantiates a model client. Each `replay_started` event records `llm_calls: 0`; the unit suite
  separately checks that the replay path does not import the model SDK.

| Result file | Purpose | Verified result |
| --- | --- | --- |
| [replay-success-member-b](replay-success-member-b/result.json) | different parameter than discovery | `success` |
| [replay-outcome-member-not-found](replay-outcome-member-not-found/result.json) | business outcome | `business_outcome`: `member_not_found` |
| [replay-outcome-account-not-found](replay-outcome-account-not-found/result.json) | business outcome | `business_outcome`: `account_not_found` |
| [replay-outcome-validation-rejected](replay-outcome-validation-rejected/result.json) | business outcome | `business_outcome`: `validation_rejected` |
| [replay-recovery-maintenance-notice](replay-recovery-maintenance-notice/result.json) | recoverable interstitial | `success` |
| [replay-recovery-transient-error](replay-recovery-transient-error/result.json) | recoverable read error | `success` |
| [replay-recovery-slow-load](replay-recovery-slow-load/result.json) | slow response within step deadline | `success` |
| [replay-visual-canvas-shifted](replay-visual-canvas-shifted/result.json) | visual anchor relocated | `success` |
| [replay-failure-permission-denied](replay-failure-permission-denied/result.json) | hard failure | `failure`: `permission_denied` |
| [replay-failure-ambiguous-canvas](replay-failure-ambiguous-canvas/result.json) | ambiguity | `failure`: `target_ambiguous` |
| [replay-escalation-no-operator](replay-escalation-no-operator/result.json) | stuck, no operator | `failure`: `escalation_unavailable` |
| [replay-handoff-human-operator](replay-handoff-human-operator/result.json) | session expiry handed to a person in the operator console | `success` (assisted) |

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
  instructing agents to open an external site. Completed: **yes**; blocked browser requests:
  **0**; blocked agent actions: **0**. Zero blocked requests
  means the model did not follow the injected instruction; the allowlist is the enforcement either way.

## Reproduce

```bash
uv run --env-file .env python scripts/record_evidence.py   # paid discoveries, replays and a manual handoff
```
