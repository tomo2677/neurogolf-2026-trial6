---
name: neurogolf-current-score-snapshot
description: Submit all currently passing NeuroGolf tasks in one official Kaggle snapshot, compare aggregate publicScore with local_points, sync task_ledger only on safe aggregate match, and record current score history; use for prompts like "現状のスコアを確認してください", "current public score", or "score snapshot".
---

# NeuroGolf Current Score Snapshot

Use this skill only when the user explicitly asks to check the current overall
official public score. This workflow spends a Kaggle submission.

For one-task official scoring, use `neurogolf-official-submit-score`. For
exactly-10 public/local synchronization, use `neurogolf-official-batch-sync`.

## Commands

Run from repo root.

Plan the snapshot:

```bash
uv run python tools/official_score_snapshot.py plan
```

Prepare a full snapshot zip:

```bash
uv run python tools/official_score_snapshot.py prepare
```

Submit and poll:

```bash
uv run python tools/official_score_snapshot.py submit --run-dir submissions/official_score_snapshot/<run_id> --confirm-submit
```

Resolve and record the snapshot:

```bash
uv run python tools/official_score_snapshot.py resolve --run-dir submissions/official_score_snapshot/<run_id>
```

## Workflow

1. Run `plan` and confirm the selected tasks are all `passes_local` rows with
   numeric `local_points`. Do not silently exclude a failing task.
2. Run `prepare`; it canonical-builds and scores every selected task with
   `--no-ledger`, then creates one `submission.zip` containing all
   `taskNNN.onnx` files at the zip root.
3. Run `submit --confirm-submit`. Respect the quota guard; if quota is skipped,
   run `resolve` to record a not-synced snapshot and stop.
4. Poll until a unique Kaggle row is attributed by `run_id`.
5. Run `resolve`. If aggregate publicScore matches the expected local total
   within `0.01`, sync all selected task official fields in `task_ledger`.
6. Always record the snapshot outcome in `official_score_snapshots.md` and
   `official_score_snapshots.json`, newest first.
7. If the aggregate score mismatches, the row is ambiguous, or the submission
   fails, do not update `task_ledger` from the snapshot.

## Rules

- Candidate tasks are all ledger rows where `status == passes_local` and
  `local_points` is numeric.
- Expected aggregate score is the sum of `local_points_at_submit`, rounded to
  2 decimals.
- Aggregate match tolerance is `abs(actual - expected) <= 0.01`.
- `scaled_public_score_400` is `actual_public_score / task_count * 400`.
- Keep score history separate from `task_ledger.md`.
- Raw artifacts stay under ignored `submissions/official_score_snapshot/<run_id>/`.
- Do not submit outside this skill, `neurogolf-official-submit-score`, or
  `neurogolf-official-batch-sync`.
