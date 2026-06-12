# task018 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 8.298108748965113 | 17925504 | 2800 | 2026-06-12T12:18:11+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | passes_local | 8.298108748965113 | 17925504 | 2800 |  | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | h001 | passes_local | 8.298108748965113 | 17925504 | 2800 | 0 | not_better | Passed but did not improve local_points. |

## Archived Summary
- None yet.
