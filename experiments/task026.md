# task026 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 18.83668519596536 | 330 | 145 | 2026-06-15T00:18:49+09:00 | exp002 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | passes_local | 18.06171551598304 | 885 | 146 | 0.0810121696981 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | threshold01-bool-output-onehot | passes_local | 18.83668519596536 | 330 | 145 | 0.774969679982 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.

## High-Upside Notes
- `threshold01-bool-output-onehot` (`0.1-1.0`): the current graph builds the compact 5x3 one-hot as `FLOAT` via `Cast` + `Sub` + `Concat`; switching that pre-pad one-hot and final output to `BOOL` should remove the float `Cast/Sub` tensors and shrink the 10-channel 5x3 pre-pad tensor while preserving scorer threshold semantics. If this fails, the lesson is that task026 still depends on float output semantics despite other low-cost tasks accepting bool one-hot outputs.
