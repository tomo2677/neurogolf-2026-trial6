---
name: neurogolf-python-generator
description: Create or repair NeuroGolf task solution files in solutions/taskNNN.py from task_specs and data, using Python only as a static ONNX graph generator with build_model() -> onnx.ModelProto.
---

# NeuroGolf Python Generator

Use this skill when creating or repairing `solutions/taskNNN.py`.

## Inputs

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

## Reattempt Policy

- If `solutions/taskNNN.py` already exists, inspect it first and repair the minimum necessary part.
- Do not silently overwrite existing work with an unrelated full rewrite.
- Future candidate-file operation is allowed, but the current canonical file is `solutions/taskNNN.py`.
- If `solutions/taskNNN.py` is missing at workflow start, it is a first-created task under `neurogolf-solve-loop`.
- For first-created tasks, failed implementations are left uncommitted in the working tree; commit only after local `passes_local` is confirmed by the solve-loop.
- For post-pass cost or `local_points` improvement, use `neurogolf-cost-experiment`; this generator skill targets canonical solution creation and repair.
- Do not use Kaggle CLI/API/browser upload or submit.
