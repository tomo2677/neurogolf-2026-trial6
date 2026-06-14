# task038 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 18.193170639607825 | 834 | 70 | 2026-06-15T00:25:43+09:00 | exp003 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |
| threshold01-bool-unary-output | impl_opt | expected_delta 0.1-1.0: keep the square-count path, but build the compact 1x5 output as `BOOL` with `Not(unary_bool)` instead of casting unary to `FLOAT` and subtracting from one. This should shrink the pre-pad tensor and remove float unary/zero-row tensors; failure would show task038 depends on float output semantics. | promoted |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | threshold01-bool-unary-output | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp002 | impl_opt | threshold01-bool-unary-output | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp003 | impl_opt | threshold01-bool-unary-output | passes_local | 18.193170639607825 | 834 | 70 | 0.183427140102 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
