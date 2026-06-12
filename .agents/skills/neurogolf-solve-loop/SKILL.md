---
name: neurogolf-solve-loop
description: Parent NeuroGolf workflow that alternates solution generation and local build/score until the task passes locally or reaches 10 attempts.
---

# NeuroGolf Solve Loop

Use this skill when asked to solve a task end-to-end.

When a task already passes and the goal is lower cost or higher `local_points`, use `neurogolf-cost-experiment` instead.

## Workflow

1. Check whether `solutions/taskNNN.py` exists before editing.
2. If it does not exist, treat the task as a `first-created task`.
3. Use `neurogolf-python-generator` to create or repair `solutions/taskNNN.py`.
4. Use `neurogolf-build-score` to build and score the task.
5. If status is `passes_local`, stop or run the first-created commit flow below.
6. If status is `fails_local`, `build_failed`, or `score_failed`, inspect `outputs/reports/taskNNN_score.json`.
7. Use `first_failure` or the build error to repair `solutions/taskNNN.py`.
8. Repeat up to 10 attempts.

## First-Created Commit Flow

For a `first-created task`, do not write failed attempts to `task_ledger.*`.

During attempts, build and score with:

```bash
uv run python tools/build_task.py --task taskNNN --no-ledger
uv run python tools/score_task.py --task taskNNN --no-ledger --report outputs/reports/taskNNN_score.json
```

Only if `outputs/reports/taskNNN_score.json` reports `status == "passes_local"`, rerun canonical build/score to update the ledger:

```bash
uv run python tools/build_task.py --task taskNNN
uv run python tools/score_task.py --task taskNNN
```

Then commit and push only task-specific tracked files:

```bash
git add solutions/taskNNN.py task_specs/taskNNN.md task_ledger.json task_ledger.md
git diff --cached --name-only
git commit -m "Add passing taskNNN solution"
git push origin main
```

Before commit, confirm `git diff --cached --name-only` contains only:

- `solutions/taskNNN.py`
- `task_specs/taskNNN.md`
- `task_ledger.json`
- `task_ledger.md`

If any other file is staged, unstage it before committing. If `git push origin main` fails, stop and do not move to the next task.

## Stop Conditions

- Stop immediately when local validation passes.
- Stop after 10 failed attempts and mark the task `needs_human_review`.
- Do not claim success unless the score report shows `passes_local`.
- For a first-created task that does not pass, leave the failed `solutions/taskNNN.py` in the working tree and do not commit it.

## Scope

- This workflow is local only.
- Do not use Kaggle CLI/API/browser upload/submit.
