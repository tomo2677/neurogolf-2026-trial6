# task012 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.045068139925302 | 7562 | 184 | 2026-06-13T04:06:29+09:00 | exp005 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | nonzero-from-color12 | passes_local | 15.547576566890212 | 12564 | 175 | 0.341751359486 | promoted | Auto promoted after canonical re-score. |
| exp002 | rule_redesign | top2-dynamic-color-slices | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp003 | rule_redesign | top2-dynamic-color-slices-equal-split | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp004 | rule_redesign | top2-dynamic-color-slices-i32-axes | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp005 | rule_redesign | top2-dynamic-color-slices-valueinfo | passes_local | 16.045068139925302 | 7562 | 184 | 0.497491573035 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
