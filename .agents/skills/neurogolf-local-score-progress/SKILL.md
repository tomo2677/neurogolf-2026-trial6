---
name: neurogolf-local-score-progress
description: Record and inspect the current local NeuroGolf score progress from task_ledger without Kaggle submission; use for prompts like "現在地を確認", "local score progress", "改善作業中の進捗確認", or "8000点までの差分".
---

# NeuroGolf Local Score Progress

Use this skill when the user asks to inspect or record the current local
score-up progress during improvement work.

This workflow is local-only. It reads `task_ledger.*` as the source of truth
and writes progress history to `local_score_progress.md` and
`local_score_progress.json`. It must not submit to Kaggle.

For official aggregate score confirmation, use
`neurogolf-current-score-snapshot` instead.

## Commands

Run from repo root.

Inspect the current local progress without changing tracked files:

```bash
uv run python tools/local_score_progress.py plan
```

Record the current local progress history:

```bash
uv run python tools/local_score_progress.py record
```

## Workflow

1. Run `plan` to inspect the current local estimate.
2. Confirm the summary comes from ledger rows with `status == "passes_local"`
   and numeric `local_points`.
3. If the user wants the current progress preserved, run `record`.
4. Check `local_score_progress.md`; it is the user-facing summary file.
5. Commit `local_score_progress.md` and `local_score_progress.json` together
   when recording progress.

## Rules

- Do not edit `task_ledger.*`; this workflow only reads it.
- Do not edit `official_score_snapshots.*`; that file is only for official
  aggregate submit history.
- `local_scaled_score_400` is the current local average scaled to 400 tasks.
- `gap_to_8000` uses the repo-internal strategic target from `AGENTS.md`.
- Keep raw/generated artifacts out of this workflow; progress history is
  tracked curated data.
