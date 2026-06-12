---
name: neurogolf-cost-experiment
description: Improve already passing or rule-invalid NeuroGolf tasks by lowering ONNX cost / raising local_points while staying public-rule compliant. Use for prompts like "cost を下げる", "passes 済みを最適化", or "task001-010 の実装を繰り返し工夫"; evaluates ignored candidate .py files and auto-promotes only better rule-compliant candidates.
---

# NeuroGolf Cost Experiment

Use this skill after a task already has `status == passes_local`, or when replacing a `rule_invalid` task with the first compliant local pass. For first-pass solving, use `neurogolf-solve-loop` instead.

Before writing candidates, read `docs/neurogolf_public_facts.md` and keep the candidate inside the public ONNX interface, file size, static-shape, banned-op, and validator constraints. The tools hard-gate violations as `rule_invalid`.

## Command

Run candidate experiments from repo root:

```bash
uv run python tools/experiment_task.py --task taskNNN --candidate outputs/experiments/taskNNN/work/candidate.py --hypothesis-id h001 --mode impl_opt
```

`--mode` must be:

- `impl_opt`: same rule/spec, cheaper ONNX implementation.
- `rule_redesign`: changed solution strategy or task rule interpretation.

The candidate must define `build_model() -> onnx.ModelProto`. Keep candidate `.py` files under `outputs/experiments/` unless the user explicitly wants a tracked file.

## Workflow

1. Read `task_ledger.json` and select `passes_local` tasks in the requested range. Include `rule_invalid` tasks only when the goal is a compliant replacement.
2. For one task at a time, read `task_specs/taskNNN.md`, `solutions/taskNNN.py`, and `experiments/taskNNN.md` if it exists.
3. Use one active hypothesis, or add one concise hypothesis to `experiments/taskNNN.md`.
4. Write an ignored candidate `.py` under `outputs/experiments/taskNNN/work/`.
5. Run `tools/experiment_task.py` with the candidate, hypothesis id, and mode.
6. Read the experiment report and the bounded note.
7. If the candidate is rule-compliant and improves `local_points`, the tool auto-promotes it by re-running canonical build/score. For a `rule_invalid` baseline, the first rule-compliant `passes_local` candidate may be promoted even without a numeric baseline score.
8. If the report has `decision == "promoted"`, run the promotion commit flow below.
9. Move to the next task or next hypothesis until a blocker or user stop condition is reached.

## Promotion Commit Flow

Commit and push only after `tools/experiment_task.py` reports `decision == "promoted"`.

For a `passes_local` baseline, the improvement must satisfy both conditions:

- Candidate report has `status == "passes_local"`.
- Canonical re-score after promotion has higher numeric `local_points` than the baseline.

For a `rule_invalid` baseline, promotion requires canonical re-score to report `status == "passes_local"` with numeric `local_points`.

Stage only the promoted task files:

```bash
git add solutions/taskNNN.py experiments/taskNNN.md task_ledger.json task_ledger.md
git diff --cached --name-only
git diff --cached --stat
git commit -m "Improve taskNNN solution cost"
git push origin main
```

Before commit, confirm `git diff --cached --name-only` contains only:

- `solutions/taskNNN.py`
- `experiments/taskNNN.md`
- `task_ledger.json`
- `task_ledger.md`

If any other file is staged, unstage it before committing. If `git push origin main` fails, stop and do not move to the next task or experiment.

Do not commit candidates that are `not_better`, tied, `build_failed`, `score_failed`, `rule_invalid`, or `promotion_failed`. Candidate `.py` files and raw reports under `outputs/experiments/` remain ignored artifacts.

## Notes Discipline

Tracked notes live at `experiments/taskNNN.md`.

- Keep `Current Best`, `Active Hypotheses`, `Experiment Log`, and `Archived Summary`.
- Keep at most 5 active hypotheses.
- Keep the log compact; the tool retains the most recent 25 rows.
- Do not paste candidate code or raw JSON reports into Markdown.
- Put raw artifacts under `outputs/experiments/taskNNN/`.
- Treat `task_specs/` as canonical rule descriptions. Do not auto-edit specs; record spec hash/dirty status through the experiment report.

## Stop Conditions

Stop only for a clear blocker or a user-specified stop condition.

Clear blockers:

- Build/score tooling is broken across multiple tasks.
- Official scoring appears unstable enough that results cannot be trusted.
- Spec and data conflict so badly that no next hypothesis can be stated.
- The same task has repeated unexplained `build_failed` candidates.
- A policy change needs user judgment.

Not blockers:

- Score does not improve.
- The implementation is hard.
- Many attempts have already been made.

## Integrity

- Do not use Kaggle CLI/API/browser upload or submit.
- Build and score one task at a time.
- Do not convert an arbitrary Python runtime solver to ONNX.
- Do not claim improvement unless the canonical re-score after promotion reports `passes_local` and higher `local_points`; for `rule_invalid` baselines, call it a compliant replacement instead.
- Do not treat `rule_invalid` as a valid best, even if it has a high local functional score.
