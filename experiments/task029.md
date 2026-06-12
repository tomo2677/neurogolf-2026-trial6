# task029 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 12.227384187753293 | 349465 | 2968 | 2026-06-13T07:55:44+09:00 | exp005 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | passes_local | 12.140805610766446 | 382403 | 1903 | 0.358293696231 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | int32-frame-indices | passes_local | 12.189474784925212 | 364147 | 1903 | 0.0486691741588 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | crop23-window | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp004 | impl_opt | crop23-window-split-grids | passes_local | 12.227381350338574 | 349465 | 2969 | 0.0379065654134 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | reuse-k1 | passes_local | 12.227384187753293 | 349465 | 2968 | 2.83741471918e-06 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
