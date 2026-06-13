# task037 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 15.330969758678355 | 15583 | 237 | 2026-06-13T17:41:49+09:00 | exp013 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |
| u8-diag-grids | impl_opt | Replace FLOAT diagonal arithmetic and point grids with static UINT8 main/anti diagonal grids to reduce dense 9x10x10 tensors. | promoted |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | public-distance6-u8-output | passes_local | 14.610943095114045 | 26400 | 6102 | 0.341102970815 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | f16-conv-scores | passes_local | 14.79263719511734 | 21000 | 6102 | 0.181694100003 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | bbox-diagonal-lines-v2 | passes_local | 14.970717151419564 | 22642 | 39 | 0.178079956302 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | float-presence | passes_local | 14.98550788942096 | 22309 | 39 | 0.0147907380014 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | f16-diag-grids | passes_local | 15.199653206383385 | 18001 | 39 | 0.214145316962 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | u8-diag-grids | passes_local | 15.330969758678355 | 15583 | 237 | 0.131316552295 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
