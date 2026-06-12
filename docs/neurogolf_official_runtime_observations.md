# NeuroGolf Official Runtime Observations

This document records team-observed official Kaggle runtime behavior that is
not yet part of `docs/neurogolf_public_facts.md`.

Do not treat these observations as public competition rules unless they are
confirmed by official competition pages, files, or host clarification.

## 2026-06-12 TopK UINT8 compatibility

Observed submissions:

- `task004` with `TopK(UINT8)` returned `SubmissionStatus.ERROR`.
- `task005` with `TopK(UINT8)` returned `SubmissionStatus.ERROR`.
- `task007` with `TopK(UINT8)` returned `SubmissionStatus.ERROR`.
- `task009` with `TopK(UINT8)` returned `SubmissionStatus.ERROR`.
- `task018` with `TopK(FLOAT)` returned `SubmissionStatus.COMPLETE`.

Local validation with the bundled utility and local ONNX Runtime accepted the
`TopK(UINT8)` models, so this appears to be an official-runtime compatibility
gap rather than a public validator finding.

Project policy:

- Use `TopK` only with `FLOAT` data input.
- If scores are logically boolean or integer, keep them as `FLOAT` until after
  `TopK`.
- Treat `TopK(non-FLOAT)` as rule-invalid in local tooling to prevent official
  `SubmissionStatus.ERROR` repeats.
