# task013 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 15.437876630165512 | 14111 | 105 | 2026-06-13T05:34:04+09:00 | exp004 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | gathernd-point-colors | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp002 | impl_opt | gathernd-point-colors-float-masks | passes_local | 14.473010136609815 | 37211 | 98 | 1.63106831704 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | u8-nonblack-masks | passes_local | 15.068508251054274 | 20471 | 97 | 0.595498114444 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | gather-extreme-points | passes_local | 15.437876630165512 | 14111 | 105 | 0.369368379111 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
