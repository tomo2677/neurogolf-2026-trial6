# task028 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.60366470619919 | 1578 | 52 | 2026-06-13T06:13:31+09:00 | exp004 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | rule_redesign | fixed-row2-row7-colors | passes_local | 17.320286360033627 | 1926 | 238 | 1.42337554126 | promoted | Auto promoted after canonical re-score. |
| exp002 | rule_redesign | cols2-7-row-scan | passes_local | 17.46310287043383 | 1638 | 238 | 0.1428165104 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | row-fragments-template | passes_local | 17.60366470619919 | 1578 | 52 | 0.140561835765 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
