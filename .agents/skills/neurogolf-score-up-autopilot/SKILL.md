---
name: neurogolf-score-up-autopilot
description: 'Run the NeuroGolf long-running score-up workflow for prompts like "スコア上げといて": fill new task baselines first, prioritize high-upside local_points hypotheses, and trigger official scoring only after meaningful local gains.'
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
   `neurogolf-official-submit-score`, respecting quota guard. This includes
   ledger statuses such as `submitted`, `pending`, `not_found`,
   `poll_failed`, and `quota_skipped`.
4. If `official_zero` is nonempty, process the lowest-numbered task with
   `neurogolf-official-zero-repair`.
5. If `score_up_candidates` is nonempty, process an existing gate-passing
   improvement with `neurogolf-official-submit-score` before starting new
   local experiments. Prefer largest `delta`, then lowest task number.
6. Otherwise run the High-Upside Hypothesis Loop below, then process one
   selected task with `neurogolf-cost-experiment`.
7. After a promoted local improvement, run:

```bash
uv run python tools/score_up_gate.py should-submit --task taskNNN
```

8. Submit only when the gate returns `can_submit: true`; use
   `neurogolf-official-submit-score`.
9. Keep moving to another task unless a clear blocker affects the overall
   toolchain or official score attribution.

## High-Upside Hypothesis Loop

Use this loop only after baseline, official-pending, official-zero, and
existing `score_up_candidates` work is exhausted.

1. Read `task_ledger.json`, relevant `experiments/taskNNN.md` notes, and
   current `solutions/taskNNN.py` files before selecting a local experiment.
2. Do not choose a `passes_local` task mechanically. Rank candidate tasks by
   credible upside using current `local_points`, `official_public_score`,
   `memory_bytes_approx`, `params`, solution complexity, past experiment
   deltas, and remaining untested strategy changes.
3. Prefer hypotheses in this fixed order:
   - `expected_delta >= 1.0`
   - `0.5 <= expected_delta < 1.0`
   - If only `< 0.5` hypotheses remain, rebuild hypotheses for each plausible
     task and look specifically for a `>= 0.5` path.
   - If no credible `>= 0.5` hypothesis remains after rebuilding, stop local
     exploration and wait for the next user prompt.
4. Treat a hypothesis as credible only when it names a specific mechanism,
   expected delta bucket, supporting evidence from the task or past logs, and
   what a failed candidate would teach.
5. Prioritize structural changes over micro cleanup: `rule_redesign`, dense
   grid replacement, crop/window/algorithm redesign, large intermediate
   removal, and alternate ONNX formulations that could materially change cost.
6. Do not make `drop-unused-*`, initializer cleanup, or other tiny changes the
   main experiment unless they follow a high-upside promotion or the user asks
   for small cleanup. If a high-upside candidate accidentally promotes only a
   small gain, accept the existing `tools/experiment_task.py` decision.
7. Record why the selected task and hypothesis beat the alternatives before
   running the candidate. Keep the note compact.

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
