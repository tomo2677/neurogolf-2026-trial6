---
name: neurogolf-official-submit-score
description: Prepare, submit, poll, and record a single NeuroGolf task's official Kaggle score after local validation, using run_id-based submission row attribution and task_ledger official columns.
---

# NeuroGolf Official Submit Score

Use this skill only when the user explicitly asks for official Kaggle scoring, hidden-score probing, or a workflow that includes submit.

This workflow spends a Kaggle submission. Default to exactly one task.

For newly added task batches or "baseline the new tasks" requests, use
`neurogolf-new-task-baseline` as the parent workflow. It calls this skill one
task at a time after local validation.

## Commands

Run from repo root.

Prepare a one-task submission zip:

```bash
uv run python tools/official_submission.py prepare --task taskNNN
```

Submit and poll for the official row:

```bash
uv run python tools/official_submission.py submit --run-dir submissions/official/taskNNN/<run_id> --confirm-submit
```

Poll an already submitted run:

```bash
uv run python tools/official_submission.py poll --run-dir submissions/official/taskNNN/<run_id>
```

Check the team daily submit quota without submitting:

```bash
uv run python tools/official_submission.py quota
```

## Workflow

1. Confirm the task is intended for official scoring and handle one task at a time.
2. Run `prepare`; it canonical-builds and scores the task locally, then creates `submission.zip` with only `taskNNN.onnx` at the zip root.
3. Read the printed `run_dir` and check `manifest.json` if needed.
4. Run `submit --confirm-submit` only when submit is intended. The script checks team daily quota immediately before submitting.
5. Accept an official score only if the Kaggle submissions CSV has exactly one row whose `description` contains the manifest `run_id`.
6. Check `task_ledger.json` and `task_ledger.md` for the updated official columns.
7. If a row returns `SubmissionStatus.ERROR`, inspect `docs/neurogolf_official_runtime_observations.md` before resubmitting.
8. If a row returns `SubmissionStatus.COMPLETE` with `publicScore == 0.0` for a
   locally passing task, treat it as hidden correctness failure and use
   `neurogolf-official-zero-repair` before spending another submit.
9. If submit returns `quota_skipped`, do not retry the same day unless the user explicitly overrides the policy in code. Resume after quota resets.

## Integrity

- Do not submit multiple tasks unless the user explicitly requests that.
- Do not use a Kaggle row without a unique `run_id` match.
- Do not put manifest files inside `submission.zip`.
- Raw artifacts stay under ignored `submissions/official/taskNNN/<run_id>/`.
- Treat Kaggle `publicScore` as authoritative for the submitted zip; local score remains a development estimate.
- This competition does not expose a usable private official score. Do not add
  `official_private_score` to the ledger; track `official_public_score` only.
- Team daily quota policy: limit is `100`; if remaining submissions are `10` or fewer, or quota cannot be checked, the script must skip submit and write `official_status=quota_skipped`.
- Known official-runtime guardrail: keep `TopK` data input as `FLOAT`.
