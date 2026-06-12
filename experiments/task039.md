# task039 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.212627011268125 | 6520 | 31 | 2026-06-13T03:39:45+09:00 | exp002 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | dynamic-slice-crop | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp002 | impl_opt | dynamic-gather-crop | passes_local | 16.212627011268125 | 6520 | 31 | 0.495567075659 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
