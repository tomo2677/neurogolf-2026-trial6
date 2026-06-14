---
name: neurogolf-official-batch-sync
description: Batch-submit exactly 10 NeuroGolf tasks in one official Kaggle submission, compare the aggregate publicScore with local_points, and sync task_ledger official_public_score only when aggregate attribution is safe; use for prompts like "10件まとめて提出", "public/local 差分をまとめて同期", or "batch sync official scores".
---

# NeuroGolf Official Batch Sync

Use this skill only when the user explicitly asks for official batch scoring or
public/local score synchronization. This workflow spends a Kaggle submission.

For one-task official scoring, use `neurogolf-official-submit-score`. For
hidden-correctness repair after a confirmed `publicScore == 0.0`, use
`neurogolf-official-zero-repair`.

## Commands

Run from repo root.

Plan the next exactly-10 batch:

```bash
uv run python tools/official_batch_sync.py plan --limit 10
```

Prepare a batch zip:

```bash
uv run python tools/official_batch_sync.py prepare --tasks taskNNN taskMMM ... --limit 10
```

Submit and poll:

```bash
uv run python tools/official_batch_sync.py submit --run-dir submissions/official_batch/<run_id> --confirm-submit
```

Resolve and sync the ledger only after a completed official row:

```bash
uv run python tools/official_batch_sync.py resolve --run-dir submissions/official_batch/<run_id>
```

If `resolve` reports `one_zero_probe_required`, confirm with the single-task
workflow, then resolve the remaining 9 tasks:

```bash
uv run python tools/official_batch_sync.py resolve --run-dir submissions/official_batch/<run_id> --zero-run-dir submissions/official/taskNNN/<run_id>
```

## Workflow

1. Run `uv run python tools/score_up_gate.py status`.
2. Run batch sync only after `baseline_targets`, `official_pending`,
   `official_zero`, and single-task `score_up_candidates` are exhausted.
3. Use `tools/official_batch_sync.py plan --limit 10`. If fewer than 10
   candidates exist, stop; do not submit a smaller batch.
4. Prepare, submit, and poll the selected 10 tasks.
5. Run `resolve`. If aggregate publicScore matches the expected local total
   within `0.01`, accept the attribution and sync all 10 tasks.
6. If one task being official zero exactly explains the aggregate deficit,
   submit that one task via `neurogolf-official-submit-score`.
7. If the single-task publicScore is `0.0`, resolve with `--zero-run-dir` and
   sync only the remaining 9 tasks. Leave the zero task for
   `neurogolf-official-zero-repair`.
8. If the mismatch is ambiguous, the one-zero probe is nonzero, or quota is
   skipped, stop and do not update the ledger from the batch.

## Rules

- Batch size is exactly 10 tasks.
- Candidate order is `local_points - official_public_score` descending, then
  task number ascending.
- Expected aggregate score is the sum of `local_points_at_submit`, rounded to
  2 decimals.
- Aggregate match tolerance is `abs(actual - expected) <= 0.01`.
- Do not write per-task official fields until `resolve` proves safe
  attribution.
- Preserve public-only score tracking; do not add `official_private_score`.
- Raw artifacts stay under ignored `submissions/official_batch/<run_id>/`.
- Do not submit outside this skill or `neurogolf-official-submit-score`.
