# NeuroGolf 2026 Public Competition Facts

Last checked: 2026-06-12

This document records public, factual information about Kaggle
`neurogolf-2026`.

This is not:

- an implementation guide
- a workflow guide
- a coding-skill document
- a strategy memo
- a task-specific solution note
- a leaderboard optimization note

The purpose of this document is to prevent humans and AI coding agents from
misreading the competition contract: data format, submission format, ONNX
interface, official constraints, validator behavior, scoring, and known public
caveats.

## 1. Source categories

Facts in this document are grouped by source type.

### 1.1 Official Kaggle competition pages

Includes public Kaggle pages such as:

- Overview / Description
- Data
- Evaluation
- Constraints
- Rules
- Timeline

These should be treated as primary competition rules.

### 1.2 Official competition files

Includes files distributed through the competition dataset, especially:

- `neurogolf_utils/neurogolf_utils.py`
- `task001.json` through `task400.json`

Facts derived from these files should be treated as official utility behavior
or public-data observations, depending on the fact.

### 1.3 Kaggle host discussions / clarifications

Host comments and update posts may clarify official behavior.

When using a host discussion as a source, record:

- date
- topic
- whether it changes, clarifies, or supersedes earlier text

### 1.4 Public-data observations

Facts obtained by inspecting the downloaded public dataset should be labeled as
observations, not as competition rules.

Examples:

- observed task count
- observed example counts
- observed grid dimensions
- observed color IDs

## 2. Competition summary

`neurogolf-2026` is a Kaggle competition based on the 400 ARC-AGI public
training v1 tasks.

For each task, participants submit a task-specific ONNX network that maps an
input grid to an output grid.

Unlike normal ARC-style submissions, the submitted artifact is not a JSON file
of predicted outputs. The submission is a zip file containing ONNX models.

The objective is twofold:

1. the network must be functionally correct
2. the network cost should be as small as possible

Only functionally correct networks receive points.

## 3. Timeline and operational rules

Official timeline as currently recorded:

- Start Date: `2026-04-15`
- Entry Deadline: `2026-07-08 23:59 UTC`
- Team Merger Deadline: `2026-07-08 23:59 UTC`
- Final Submission Deadline: `2026-07-15 23:59 UTC`

Known operational facts / notes:

- Competition category: `Research`
- Evaluation metric name: `NeuroGolf Metric`
- Prize pool: `$50,000`
- Maximum team size: `5`
- Final submissions selected for judging: up to `2`
- Winner license type: Open Source, Apache 2.0
- Competition data license/use: Competition Use and Commercial, Apache 2.0

Submission limit caveat:

- Kaggle metadata and host update have indicated `100` submissions per day.
- Some older rule text may still mention `5` submissions per day.
- Before high-volume submissions or final submission, re-check the current
  Kaggle UI/API and official announcements.

Sharing / external resources:

- Private code/data sharing outside the team is not allowed.
- Public sharing is allowed when made available to all participants through
  Kaggle competition forums/notebooks and compatible with the licensing rules.
- External data/tools are allowed unless specifically prohibited by the host,
  but they must be publicly available or reasonably accessible and must not
  create unfair access barriers.

## 4. Data

The competition data is distributed as `neurogolf-2026.zip`.

The downloaded competition data includes:

- `neurogolf_utils/neurogolf_utils.py`
- `task001.json` through `task400.json`

Each `taskNNN.json` file is a dictionary with these keys:

- `train`
- `test`
- `arc-gen`

Each example has this structure:

```json
{
  "input": [[0, 1, 2], [3, 4, 5]],
  "output": [[...]]
}
```

Grid facts:

- A grid is a rectangular matrix.
- Cell values are color IDs.
- Color IDs are integers `0` through `9`.

Observed public-data facts from the downloaded dataset:

- number of tasks: `400`
- all tasks have `train`, `test`, and `arc-gen`
- observed colors: `0` through `9`
- extracted grids are rectangular
- `train` and `test` grids are within `30x30`
- public `arc-gen` contains some examples larger than `30x30`
- official utility behavior ignores examples whose largest input/output grid
  dimension exceeds `30`

Important caveat:

The data-description may describe grid size as up to `30x30`, but the public
`arc-gen` data has included examples larger than `30x30`. Treat such examples
according to the official utility behavior: examples outside the supported
`30x30` envelope are ignored for local verification/scoring.

## 5. Submission format

The submission file should be named:

```text
submission.zip
```

The zip contains at most one ONNX file per task.

Expected file names:

```text
task001.onnx
task002.onnx
...
task400.onnx
```

Submission facts / assumptions:

- A task may be omitted.
- An omitted task receives no points.
- Each task can have at most one ONNX file.
- The safest zip layout is to place `taskNNN.onnx` files at the root of the zip.
- Avoid nested directories unless official text explicitly confirms they are
  accepted.
- Generated submission artifacts are normally not suitable for Git tracking.

## 6. ONNX model interface

Official utility interface:

- input tensor name: `input`
- output tensor name: `output`
- input tensor shape: `[1, 10, 30, 30]`
- output tensor shape: `[1, 10, 30, 30]`
- dtype used by the official helper: `FLOAT`
- channel dimension: 10 color channels

Color/channel convention:

- channel `0` corresponds to color ID `0`
- channel `1` corresponds to color ID `1`
- ...
- channel `9` corresponds to color ID `9`

Input encoding:

- The original grid is top-left anchored.
- For a cell `(r, c)` with color `k`, the input tensor has:

```text
input[0, k, r, c] = 1.0
```

- Cells outside the original grid are zero-hot.
- Zero-hot means all 10 channels are `0.0`.
- Padded cells are not background color `0`; they are zero-hot.

Output interpretation:

- The network output is thresholded with `> 0.0`.
- The thresholded output is treated as a binary tensor.
- Expected output is also represented as a one-hot/zero-hot tensor.
- Correctness requires exact tensor equality.
- Cells outside the expected output grid must be zero-hot.

## 7. Official ONNX constraints

Official competition constraints include:

- All tensors and parameters must have statically-defined shapes.
- Each ONNX file must be at most `1.44 MB`.
- Disallowed ONNX operations:

```text
Loop
Scan
NonZero
Unique
Script
Function
```

These are the official banned operations listed in the competition constraints.

## 8. Official utility / validator behavior

The public official utility is stricter than the short official banned-op list.

As of the currently checked utility behavior, models should avoid or expect
rejection for the following:

- `Compress`
- any op type containing `Sequence`
- custom domains
- domains other than default ONNX domain `""` or `ai.onnx`
- model functions
- subgraphs
- multi-input graphs
- multi-output graphs
- sequence tensors
- symbolic shapes / `dim_param`
- missing `dim_value`
- zero or negative dimensions
- tensor / initializer name collisions
- duplicate `value_info` names
- tensor names containing ONNX Runtime profiler's `kernel_time` marker
- models that cannot be loaded by ONNX Runtime after sanitization
- models whose performance cannot be measured

Important distinction:

The official banned-op list and the official utility's stricter behavior are
not the same thing.

Keep these categories separate:

```text
Official constraints page:
- Loop
- Scan
- NonZero
- Unique
- Script
- Function
- static shapes
- 1.44 MB file limit

Official utility / validator behavior:
- additional rejection behavior such as Compress, Sequence*, custom domains,
  subgraphs, non-static shapes, multi-input/output graphs, and profiler/scoring
  failures
```

## 9. Static-shape rule

All tensor shapes must be statically defined.

Do not emit:

- symbolic dimensions
- `dim_param`
- missing dimension values
- zero dimensions
- negative dimensions
- runtime-dependent output shapes
- dynamically-sized intermediate tensors
- data-dependent shape-changing graphs

Data-dependent tensor values are not inherently forbidden, but tensor shapes
must remain statically defined and acceptable to the official utility.

Practical interpretation:

- Dynamic control flow should be avoided.
- Dynamic shape construction should be avoided.
- Shape inference should be able to determine all tensor shapes.
- ONNX Runtime should be able to load the sanitized model.

## 10. Scoring

Official per-task scoring formula:

```text
points = max(1, 25 - ln(cost))
```

Only functionally correct networks receive points.

Incorrect networks receive no points for that task.

Current cost interpretation from official updates / utility behavior:

```text
cost = total_parameters + total_memory_footprint_bytes
```

Important scoring facts:

- Higher score is better.
- Smaller cost gives more points.
- The theoretical maximum for one task is `25` points.
- The theoretical maximum over 400 tasks is `10000` points.
- Current utility behavior does not count MACs in cost.
- Parameters and constant values can contribute to cost.
- Input/output tensors named `input` and `output` are excluded from memory
  footprint accounting by the official utility.

Local utility formula observed in the public utility:

```text
points = max(1.0, 25.0 - ln(max(1.0, memory + params)))
```

`memory` is based on intermediate tensor memory footprint.

`params` includes, for example:

- dense initializers
- sparse initializer values
- `Constant` tensor values
- `Constant` attribute lists such as `value_floats`, `value_ints`, and
  `value_strings`

Caveat:

Some older descriptions or summaries may mention MACs. Treat the current
Kaggle scorer, current official utility, and host update as authoritative for
the active competition.

## 11. Functional correctness and hidden benchmark

Functional correctness is checked against public benchmarks and a private
benchmark suite.

Public/local validation can check only public examples:

- `train`
- `test`
- `arc-gen`

It cannot reproduce hidden benchmark checks.

Host clarification has indicated:

- the current leaderboard already incorporates the hidden benchmark suite
- no separate post-competition leaderboard is expected
- hidden tests use the same top-left anchored `[1, 10, 30, 30]` tensor format
- hidden tests are constrained to grids of size `30x30` or smaller

Important caveat:

Local score is useful for development, but official Kaggle scorer / leaderboard
results are authoritative.

A model that passes public local validation can still fail hidden checks.

## 12. Public-data caveats

Known public-data caveats:

- Public `arc-gen` includes some examples larger than `30x30`.
- Official utility ignores examples outside the supported `30x30` envelope.
- Local public validation cannot prove hidden correctness.
- Official scorer behavior may be updated during the competition.
- Submission limits and operational rules may change or be clarified.
- Re-check Kaggle UI/API and official announcements before final submission.
- If local score and Kaggle leaderboard disagree, treat Kaggle as
  authoritative.

## 13. Do not put implementation strategy here

This document should not contain:

- task-specific solution ideas
- heuristic catalogs
- generated ONNX recipes
- model construction code
- scoring scripts
- workflow instructions
- CI instructions
- leaderboard strategy
- local experiment logs
- private team notes

Those belong in separate implementation, workflow, or experiment documents.

This document should remain a public-facts reference for the competition
contract.

## 14. Minimal checklist for AI coding agents

Before generating or modifying an ONNX model, remember:

- The model is task-specific.
- The model must have one input named `input`.
- The model must have one output named `output`.
- Input shape is `[1, 10, 30, 30]`.
- Output shape is `[1, 10, 30, 30]`.
- Tensor dtype should be compatible with the official `FLOAT` interface.
- Input grids are one-hot encoded and top-left anchored.
- Padded input cells are zero-hot.
- Output is thresholded with `> 0.0`.
- Padded output cells must be zero-hot.
- All shapes must be static.
- Each ONNX file must be at most `1.44 MB`.
- Do not use official banned ops.
- Also avoid utility-rejected constructs such as `Compress`, `Sequence*`,
  custom domains, subgraphs, sequence tensors, symbolic shapes, and
  multi-input/output graphs.
- Passing public local validation is necessary but not sufficient for official
  correctness.
- Kaggle scorer / leaderboard is authoritative.

## 15. Source log

Checked date: 2026-06-12

Official competition pages to keep in sync:

- Kaggle Overview / Description
- Kaggle Data
- Kaggle Evaluation
- Kaggle Constraints
- Kaggle Rules
- Kaggle Timeline

Official files to keep in sync:

- `neurogolf_utils/neurogolf_utils.py`
- `task001.json` through `task400.json`

Host updates / clarifications to keep in sync:

- 2026-04-21 update:
  - submission limit increased to `100/day`
  - examples larger than `30x30` ignored by metric
  - environment/version guidance
- 2026-05-14 utility release notes:
  - stricter sanitizer / validator behavior
- 2026-06-10 clarification:
  - current leaderboard already includes hidden benchmark suite
  - no separate post-deadline leaderboard expected
  - hidden examples use the same tensor format
  - hidden grids are `30x30` or smaller

Local public-data observations to keep in sync:

- task count
- example counts
- observed grid dimensions
- observed colors
- presence of `>30x30` examples in public `arc-gen`