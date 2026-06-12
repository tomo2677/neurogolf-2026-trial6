---
name: neurogolf-python-generator
description: Create or repair NeuroGolf task solution files in solutions/taskNNN.py from task_specs and data, using Python only as a static ONNX graph generator with build_model() -> onnx.ModelProto.
---

# NeuroGolf Python Generator

Use this skill when creating or repairing `solutions/taskNNN.py`.

## Inputs

- Before creating or modifying ONNX generator code, read `docs/neurogolf_public_facts.md`, especially ONNX interface, constraints, validator behavior, and the minimal checklist.
- Read `task_specs/taskNNN.md`; if missing, also check `task_specs/task_N.md`.
- Read `data/taskNNN.json` and inspect `train`, `test`, and `arc-gen`.
- Ignore examples where input or output exceeds 30x30.

## Contract

- `solutions/taskNNN.py` must define `build_model() -> onnx.ModelProto`.
- ONNX input name: `input`.
- ONNX output name: `output`.
- Input/output shape: `FLOAT[1,10,30,30]`.
- Input grid is top-left anchored one-hot; padded area is zero-hot.
- Network output is thresholded with `> 0.0` before exact-match validation.
- Use static shapes.

## Design Rules

- Do not implement an arbitrary Python runtime solver.
- Python is only the ONNX graph generator.
- Prefer small static ONNX graphs using standard ONNX ops.
- Avoid `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`, and `Sequence` ops.
- Use `TopK` only with `FLOAT` data input; see `docs/neurogolf_official_runtime_observations.md` for the official-runtime `TopK(UINT8)` failure observation.
- Keep the generated model rule-compliant before optimizing score: file size must stay within `1.44 MB`, graph shapes must be static, and utility-rejected constructs from `docs/neurogolf_public_facts.md` must be avoided.

## Reattempt Policy

- If `solutions/taskNNN.py` already exists, inspect it first and repair the minimum necessary part.
- Do not silently overwrite existing work with an unrelated full rewrite.
- Future candidate-file operation is allowed, but the current canonical file is `solutions/taskNNN.py`.
- If `solutions/taskNNN.py` is missing at workflow start, it is a first-created task under `neurogolf-solve-loop`.
- For first-created tasks, failed implementations are left uncommitted in the working tree; commit only after local `passes_local` is confirmed by the solve-loop.
- For post-pass cost or `local_points` improvement, use `neurogolf-cost-experiment`; this generator skill targets canonical solution creation and repair.
- Do not submit to Kaggle from this generator workflow; use `neurogolf-official-submit-score` only when official scoring is explicitly requested.
