# task015 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.5810813778521 | 4464 | 68 | 2026-06-13T05:05:31+09:00 | exp006 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | nonblack-from-color9 | passes_local | 16.134688367328152 | 6975 | 107 | 0.377302299855 | promoted | Auto promoted after canonical re-score. |
| exp002 | rule_redesign | four-color-direct-grid | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp003 | rule_redesign | four-color-direct-add-grid | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp004 | rule_redesign | four-color-direct-add-opset14 | passes_local | 16.53578573337465 | 4626 | 116 | 0.401097366046 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | four-color-direct-add-clean | passes_local | 16.53641857803241 | 4626 | 113 | 0.000632844657762 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | h006 | passes_local | 16.5810813778521 | 4464 | 68 | 0.0446627998197 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
