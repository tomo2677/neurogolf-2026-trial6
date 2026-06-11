---
name: neurogolf-solve-loop
description: Parent NeuroGolf workflow that alternates solution generation and local build/score until the task passes locally or reaches 10 attempts.
---

# NeuroGolf Solve Loop

Use this skill when asked to solve a task end-to-end.

When a task already passes and the goal is lower cost or higher `local_points`, use `neurogolf-cost-experiment` instead.

## Workflow

1. Use `neurogolf-python-generator` to create or repair `solutions/taskNNN.py`.
2. Use `neurogolf-build-score` to build and score the task.
3. If status is `passes_local`, stop.
4. If status is `fails_local`, `build_failed`, or `score_failed`, inspect `outputs/reports/taskNNN_score.json`.
5. Use `first_failure` or the build error to repair `solutions/taskNNN.py`.
6. Repeat up to 10 attempts.

## Stop Conditions

- Stop immediately when local validation passes.
- Stop after 10 failed attempts and mark the task `needs_human_review`.
- Do not claim success unless the score report shows `passes_local`.

## Scope

- This workflow is local only.
- Do not use Kaggle CLI/API/browser upload/submit.
