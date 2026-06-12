---
name: neurogolf-score-up-autopilot
description: Run the NeuroGolf long-running score-up workflow for prompts like "スコア上げといて": fill new task baselines first, improve local_points one task at a time, and trigger official scoring only after meaningful local gains.
---

# NeuroGolf Score-Up Autopilot

Use this skill for requests such as "スコア上げといて", "長時間スコア改善して",
or "autonomous score-up".

This is a parent workflow. It delegates local solving, local cost experiments,
official submit/poll, and official-zero repair to the smaller skills.

## Priority Loop

1. Run `uv run python tools/score_up_gate.py status`.
2. If `baseline_targets` is nonempty, process the lowest-numbered task first
   with `neurogolf-new-task-baseline`.
3. If `official_pending` is nonempty, resume the lowest-numbered task with
   `neurogolf-official-submit-score`, respecting quota guard.
4. If `official_zero` is nonempty, process the lowest-numbered task with
   `neurogolf-official-zero-repair`.
5. Otherwise select one `passes_local` task for `neurogolf-cost-experiment`.
6. After a promoted local improvement, run:

```bash
uv run python tools/score_up_gate.py should-submit --task taskNNN
```

7. Submit only when the gate returns `can_submit: true`; use
   `neurogolf-official-submit-score`.
8. Keep moving to another task unless a clear blocker affects the overall
   toolchain or official score attribution.

## Score-Up Submit Gate

- If `local_points < 20`, submit only when
  `local_points - official_public_score >= 2.0`.
- If `local_points >= 20`, submit only when
  `local_points - official_public_score >= 1.0`.
- Do not submit if `official_status != "complete"` or
  `official_public_score` is missing.
- If `official_public_score == 0.0`, do not run cost optimization first; use
  `neurogolf-official-zero-repair`.

## Handling Outcomes

- `complete` with nonzero public score close to local points: commit and push
  the verified improvement.
- `complete` with `publicScore == 0.0`: record a hidden-correctness hypothesis
  and enter `neurogolf-official-zero-repair`.
- `SubmissionStatus.ERROR`, ambiguous row, or unmatched row: stop only that task,
  record pending/investigation context, and continue other tasks.
- `quota_skipped`: keep local-only work moving; resume official scoring after
  quota resets.

## Commit / Push Policy

- During normal `main` operation, push after each completed baseline or
  official-verified improvement.
- During a feature-branch batch, make separate commits for workflow changes,
  each new-task baseline, and each official-verified improvement.
- Stage only task-relevant tracked files for task commits:

```bash
git add solutions/taskNNN.py task_specs/taskNNN.md experiments/taskNNN.md task_ledger.json task_ledger.md
```

## Integrity

- Process one task at a time.
- Do not submit outside `neurogolf-official-submit-score`.
- Preserve public-only official score tracking; do not add
  `official_private_score`.
- Do not treat a promoted local improvement as verified until official scoring
  confirms it or quota explicitly skips it.
