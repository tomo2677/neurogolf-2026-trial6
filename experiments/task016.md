# task016 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 18.65964069627225 | 540 | 27 | 2026-06-15T00:20:18+09:00 | exp002 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | channel-permute-direct | passes_local | 18.38393481486718 | 720 | 27 | 0.61304869266 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | threshold01-bool-permute-output | passes_local | 18.65964069627225 | 540 | 27 | 0.275705881405 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.

## High-Upside Notes
- `threshold01-bool-permute-output` (`0.1-1.0`): the current 3x3 channel permutation keeps both the sliced one-hot and gathered result as `FLOAT`; casting the compact slice to `BOOL` before `Gather` and making the final output `BOOL` should shrink the gathered tensor from 360 bytes to 90 bytes while adding only a 90-byte cast. If this fails, the lesson is that task016's permuted output still needs float one-hot semantics.
