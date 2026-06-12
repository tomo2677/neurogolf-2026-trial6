---
name: neurogolf-new-task-baseline
description: Fill the NeuroGolf ledger for newly added tasks by solving one task at a time, getting local and official scores, and committing each completed task.
---

# NeuroGolf New Task Baseline

Use this skill for requests such as "追加したタスクの表を埋めて",
"新しいタスクでベースライン出しといて", or "new tasks official score".

This is an end-to-end workflow. It combines local solving, local scoring,
single-task official submission, ledger update, and task-level commits.
When invoked by `neurogolf-score-up-autopilot`, return control to the parent
workflow after each completed or pending task instead of stopping the whole run.

## Selection

1. Read `task_specs/task*.md` and `task_ledger.json`.
2. Select tasks whose spec exists and either:
   - no ledger row exists,
   - `status != "passes_local"`, or
   - `official_status` is empty.
3. Process tasks in ascending task number.
4. Do not solve or submit tasks in parallel.

## Per-Task Workflow

For each task:

1. Use `neurogolf-solve-loop` to create or repair `solutions/taskNNN.py`.
2. During attempts, use `--no-ledger`; update the ledger only after local
   `passes_local`.
3. Use `neurogolf-official-submit-score` to prepare, inspect, submit, poll, and
   record one official row for that task.
4. Accept official results only by unique `run_id` match.
5. If submit returns `quota_skipped`, keep the task as local-pass with official
   pending and resume it after quota resets.
6. If official status is `SubmissionStatus.ERROR`, stop before any additional
   submit for that task, record pending/investigation context, and let the
   parent autopilot continue another task.
7. If official public score is `0.0`, use `neurogolf-official-zero-repair` and
   retry with one recorded hidden-correctness hypothesis per resubmit.
8. Commit the completed task before moving to the next task.

## Commit / Push Policy

During normal operation on `main`, push after each completed task.

For a feature-branch batch, make one task-specific commit per task and merge the
branch after all selected tasks complete.

Stage only relevant tracked files:

```bash
git add solutions/taskNNN.py task_specs/taskNNN.md task_ledger.json task_ledger.md
```

If a repair note was created or updated:

```bash
git add experiments/taskNNN.md
```

Skill or workflow updates should be committed separately from task solution
commits.

## Integrity

- Do not submit more than one task at a time.
- Follow the official submit quota guard. If remaining daily team submissions
  are `10` or fewer, or quota cannot be checked, do not submit.
- Do not continue after an unmatched, ambiguous, or errored official row.
- Do not claim completion unless `task_ledger.*` contains both local and
  official columns for the task.
- Official score tracking is public-score only for this competition; do not
  introduce an `official_private_score` ledger column.
- Keep generated artifacts under ignored directories.
