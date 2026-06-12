# task017 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 12.861970676921189 | 64641 | 122203 | 2026-06-13T05:57:02+09:00 | exp005 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | rule_redesign | h001 | passes_local | 12.780966707682527 | 78199 | 124410 |  | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | f16-period-matmul | passes_local | 12.836860951855012 | 67185 | 124410 | 0.0558942441725 | promoted | Auto promoted after canonical re-score. |
| exp003 | rule_redesign | periods-4-6-7-9 | fails_local | 0.0 | 53478 | 82068 | -12.8368609519 | fails_local | Candidate did not pass local validation. |
| exp004 | rule_redesign | periods-4-5-6-7-9 | fails_local | 0.0 | 57597 | 93536 | -12.8368609519 | fails_local | Candidate did not pass local validation. |
| exp005 | rule_redesign | periods-no-2 | passes_local | 12.861970676921189 | 64641 | 122203 | 0.0251097250662 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
