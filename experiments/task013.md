# task013 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 15.547341097323567 | 12647 | 95 | 2026-06-13T12:04:12+09:00 | exp010 |

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
| exp005 | impl_opt | dynamic-slice-colors | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp006 | impl_opt | dynamic-slice-colors-shapes | passes_local | 15.47499488424215 | 13599 | 99 | 0.0371182540766 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | reuse-shape1 | passes_local | 15.475067890265178 | 13599 | 98 | 7.30060230278e-05 | promoted | Auto promoted after canonical re-score. |
| exp008 | impl_opt | no-axes-slices | passes_local | 15.475433000349923 | 13599 | 93 | 0.000365110084745 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | int32-period-math | passes_local | 15.547262619787722 | 12647 | 96 | 0.0718296194378 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | remove-unused-two-i64 | passes_local | 15.547341097323567 | 12647 | 95 | 7.84775358458e-05 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
