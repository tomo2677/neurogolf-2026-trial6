# task011 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.1195736557076 | 2536 | 109 | 2026-06-13T05:18:59+09:00 | exp003 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | dynamic-gathernd-selected-block | passes_local | 16.519885816825184 | 4696 | 122 | 0.21338249271 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | dynamic-slice-selected-block | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp003 | impl_opt | dynamic-slice-selected-block | passes_local | 17.1195736557076 | 2536 | 109 | 0.599687838882 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
