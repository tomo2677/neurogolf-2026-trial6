---
name: neurogolf-build-score
description: Build a NeuroGolf solution Python generator into ONNX and score it locally with train, test, and arc-gen examples, updating reports and the minimal task ledger.
---

# NeuroGolf Build Score

Use this skill after editing `solutions/taskNNN.py`.

For ignored candidate files or post-pass cost experiments, use `neurogolf-cost-experiment` and `tools/experiment_task.py` instead.

## Commands

Run from repo root:

```bash
uv run python tools/build_task.py --task taskNNN
uv run python tools/score_task.py --task taskNNN
```

Canonical build/score updates `task_ledger.*` and should be used after a candidate is known to pass.

For first-created solve-loop attempts, avoid writing failed statuses to the ledger:

```bash
uv run python tools/build_task.py --task taskNNN --no-ledger
uv run python tools/score_task.py --task taskNNN --no-ledger --report outputs/reports/taskNNN_score.json
```

## Checks

- Confirm `outputs/onnx/taskNNN.onnx` exists after build.
- Confirm `outputs/reports/taskNNN_score.json` exists after score.
- Read the report for:
  - `status`
  - `local_points`
  - `memory_bytes_approx`
  - `params`
  - `first_failure`
- Treat only `status == "passes_local"` in the report as success.

## Integrity

- Do not fake success.
- If dependencies are missing, run `uv sync`.
- If scoring fails, inspect the first failure before editing the solution again.
