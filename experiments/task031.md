# task031 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 15.83727548196548 | 9205 | 330 | 2026-06-13T07:21:54+09:00 | exp006 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | passes_local | 13.77722632138248 | 72992 | 1823 | 1.30098876995 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | int32-index-path | passes_local | 14.169637922050072 | 48708 | 1824 | 0.392411600668 | promoted | Auto promoted after canonical re-score. |
| exp003 | rule_redesign | window12-crop | passes_local | 15.395322877703496 | 14508 | 326 | 1.22568495565 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | mask-fgcolor-crop | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp005 | impl_opt | mask-fgcolor-crop-zero | passes_local | 15.83717061069487 | 9205 | 331 | 0.441847732991 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | remove-unused-zero-i64 | passes_local | 15.83727548196548 | 9205 | 330 | 0.000104871270612 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
