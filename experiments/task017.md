# task017 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 13.328943217037034 | 114087 | 3045 | 2026-06-13T08:05:40+09:00 | exp009 |

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
| exp007 | impl_opt | gather-group-counts | passes_local | 12.787910492144995 | 191469 | 9738 | -0.0740601847762 | not_better | Passed but did not improve local_points. |
| exp008 | impl_opt | slice-step-counts | passes_local | 13.322306137483405 | 114087 | 3825 | 0.460335460562 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | dedupe-initializers | passes_local | 13.328943217037034 | 114087 | 3045 | 0.00663707955363 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
