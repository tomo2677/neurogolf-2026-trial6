# task039 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.8628966103607 | 3384 | 35 | 2026-06-13T03:57:31+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | dynamic-slice-crop | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp002 | impl_opt | dynamic-gather-crop | passes_local | 16.212627011268125 | 6520 | 31 | 0.495567075659 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | direct-gathernd-crop | passes_local | 16.8628966103607 | 3384 | 35 | 0.650269599093 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | gathernd-int32-indices | build_failed |  |  |  |  | build_failed | Candidate did not build. |

## Archived Summary
- None yet.
