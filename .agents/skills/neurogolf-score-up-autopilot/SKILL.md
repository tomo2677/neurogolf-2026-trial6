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
6. If `batch_sync_candidates` has at least 10 entries, process the top 10 with
   `neurogolf-official-batch-sync` before starting new local experiments.
7. Otherwise run the High-Upside Hypothesis Loop below, then process one
   selected task with `neurogolf-cost-experiment`.
8. After a promoted local improvement, run:

```bash
uv run python tools/score_up_gate.py should-submit --task taskNNN
```

9. Submit only when the gate returns `can_submit: true`; use
   `neurogolf-official-submit-score`.
10. Keep moving to another task unless a clear blocker affects the overall
   toolchain or official score attribution.

## Long-Running Iteration Contract

For autonomous score-up requests, treat the workflow as a repeated loop, not a
single experiment.

- After each completed unit of work, rerun
  `uv run python tools/score_up_gate.py status` before choosing the next unit.
- A completed unit can be a baseline task, resumed official poll, zero repair
  step, batch sync attempt, promoted cost experiment, or failed local
  experiment with notes recorded.
- Treat promotion, task commit, local progress commit, note-only commit, and a
  clean worktree as checkpoints, not stop reasons.
- After each checkpoint, return to the priority loop and re-evaluate official
  queues, batch sync eligibility, and local hypothesis ranking.
- Re-rank local experiment candidates after every ledger-changing commit and
  after every failed-probe note commit. Do not keep using a stale candidate
  order when `task_ledger.*`, `experiments/taskNNN.md`, or relevant solutions
  have changed.
- `should-submit: false`, fewer than 10 `batch_sync_candidates`, and empty
  official queues are not local score-up stop reasons. Continue into local
  experiments when no higher-priority official work is actionable.
- Continue iterating until the user-specified budget is reached, a clear
  blocker appears, official attribution becomes unsafe, quota policy requires a
  stop, or no credible `expected_delta >= 0.1` hypothesis remains after
  rebuilding hypotheses.
- When a loop stops, leave tracked current-run work clean when the commit policy
  allows it, and summarize the latest status, commits, remaining candidates,
  and stop reason.

## Autonomous Stop Reasons

For autonomous score-up runs, stop only for one of these reasons:

- The user-specified budget or stop condition is reached.
- Build/score tooling is broken across tasks.
- `git push origin main` fails after a required commit.
- Official score attribution is unsafe or quota policy requires pausing
  official work.
- The same task has repeated unexplained `build_failed` candidates.
- No credible `expected_delta >= 0.1` hypothesis remains across plausible
  tasks after rebuilding hypotheses from ledger, notes, and solutions.

Do not report the workflow as complete merely because a checkpoint was reached.
If execution must end because of context or runtime boundaries, call it a
resumable pause and report the latest `score_up_gate.py status`, the next task
or hypothesis to inspect, and any remaining stop-condition uncertainty.

## High-Upside Hypothesis Loop

Use this loop only after baseline, official-pending, official-zero,
single-task `score_up_candidates`, and batch sync work are exhausted.

1. Read `task_ledger.json`, relevant `experiments/taskNNN.md` notes, and
   current `solutions/taskNNN.py` files before selecting a local experiment.
2. Do not choose a `passes_local` task mechanically. Rank candidate tasks by
   credible local upside using current `local_points`, `memory_bytes_approx`,
   `params`, solution complexity, past experiment deltas, and remaining
   untested strategy changes.
3. Prefer hypotheses in this fixed order:
   - `expected_delta >= 1.0`
   - `0.1 <= expected_delta < 1.0`
   - If only `< 0.1` hypotheses remain, rebuild hypotheses for each plausible
     task and look specifically for a `>= 0.1` path.
   - If no credible `>= 0.1` hypothesis remains after rebuilding, stop local
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

- Submit a previously scored task one-by-one only when
  `local_points - official_public_score >= 3.0`.
- For smaller nonzero public/local gaps, wait for
  `neurogolf-official-batch-sync` to aggregate exactly 10 tasks.
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

## Local Progress Recording

When this parent workflow causes `task_ledger.*` to change, record the current
local score progress through `neurogolf-local-score-progress` after the
ledger-changing commit/push finishes.

This applies after:

- A promoted local cost improvement updates and commits `task_ledger.*`.
- A completed new-task baseline updates and commits `task_ledger.*`.
- `neurogolf-official-submit-score`, `neurogolf-official-batch-sync`, or
  `neurogolf-official-zero-repair` updates and commits official ledger fields.
- Any other score-up activity changes `task_ledger.json` or `task_ledger.md`.

Run:

```bash
uv run python tools/local_score_progress.py record
```

If `record` changes `local_score_progress.md` or `local_score_progress.json`,
commit and push those files separately:

```bash
git add local_score_progress.md local_score_progress.json
git diff --cached --name-only
git diff --cached --stat
git commit -m "Record local score progress"
git push origin main
```

Do not mix `local_score_progress.*` with task solution commits, official sync
commits, or note-only commits. If `record` produces no diff, skip the progress
commit. Do not embed this side effect in `neurogolf_onnx.update_ledger`,
`tools/build_task.py`, or `tools/score_task.py`; this parent workflow delegates
explicitly to `neurogolf-local-score-progress` to keep ledger source-of-truth
updates separate from curated progress history.

## No Lingering Dirty Notes

`tools/experiment_task.py` updates tracked `experiments/taskNNN.md` notes even
for `not_better`, `fails_local`, `build_failed`, and `rule_invalid` probes.
Keep those failed-probe lessons, but do not mix them into a promoted task
commit.

- If a task promotes, first commit and push only the promoted task files using
  the task commit policy above.
- After the promoted commit, or after a score-up run stops without promotion,
  inspect `git status --short --branch`.
- If the only remaining current-run changes are non-promoted
  `experiments/taskNNN.md` notes, stage exactly those files and commit:

```bash
git add experiments/taskAAA.md experiments/taskBBB.md
git diff --cached --name-only
git diff --cached --stat
git commit -m "Record score-up experiment notes"
git push origin main
```

- A note-only commit must not include `solutions/`, `task_ledger.*`,
  `task_specs/`, or generated artifacts.
- If unrelated dirty files already exist, leave them untouched and report them
  separately; do not hide current-run note changes behind unrelated work.
- Before the final response, run `git status --short --branch` and make the
  current-run tracked work clean whenever the commit policy allows it.
- In the final response, report task solution commits, official sync commits,
  note-only commits, and local progress commits as separate outcomes.

## Integrity

- Process one task at a time.
- Do not submit outside `neurogolf-official-submit-score`.
- Preserve public-only official score tracking; do not add
  `official_private_score`.
- Do not treat a promoted local improvement as verified until official scoring
  confirms it or quota explicitly skips it.
