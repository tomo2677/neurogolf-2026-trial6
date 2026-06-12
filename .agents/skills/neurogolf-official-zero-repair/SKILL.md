---
name: neurogolf-official-zero-repair
description: Repair a NeuroGolf task that passes local validation but receives official public score 0.0, focusing on hidden correctness generalization before cost optimization.
---

# NeuroGolf Official Zero Repair

Use this skill when `task_ledger.json` shows `status == "passes_local"`,
`official_status == "complete"`, and `official_public_score == 0.0`.

This workflow spends Kaggle submissions. The default is one task at a time and
one hypothesis per resubmit. If the user permits unlimited retries, still submit
only after recording a new hidden-correctness hypothesis.

## Workflow

1. Read `task_ledger.json`, `task_specs/taskNNN.md`, `solutions/taskNNN.py`,
   `experiments/taskNNN.md`, and the official run manifest under
   `submissions/official/taskNNN/`.
2. Treat official `0.0` as hidden correctness failure unless the row status is
   `SubmissionStatus.ERROR`.
3. Look for public-data shortcuts in the current solution:
   - fixed public max grid size instead of full `30x30`
   - fixed flood depth or scan sequence derived from public examples
   - assuming square grids when the task rule only says rectangular
   - hard-coded colors, counts, positions, or object limits not required by the
     task rule
4. Record one concise `rule_redesign` hypothesis in `experiments/taskNNN.md`.
5. Repair `solutions/taskNNN.py` for hidden correctness first. Accept lower
   `local_points` if the implementation is more faithful to the task rule.
6. Run canonical build/score and public-rules validation.
7. Use `neurogolf-official-submit-score` to prepare, inspect zip contents,
   submit, and poll the single task.
8. Accept success only when the official row is uniquely matched by `run_id`,
   `official_status == "complete"`, `official_public_score > 0.0`, and the
   rounded official score is close to local points.
9. If official score is still `0.0`, update the experiment note with the failed
   hypothesis and repeat with a new hidden-correctness hypothesis.

## Integrity

- Do not optimize cost until official `0.0` is resolved.
- Do not reuse an official row that does not uniquely match the current
  `run_id`.
- Do not mix team/private observations into `docs/neurogolf_public_facts.md`.
- If the row status is `SubmissionStatus.ERROR`, switch to official runtime
  investigation and read `docs/neurogolf_official_runtime_observations.md`.
