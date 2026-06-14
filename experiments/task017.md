# task017 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 14.920712107660183 | 20476 | 3368 | 2026-06-13T17:31:34+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |
| conv-color-map | impl_opt | Replace full-grid ArgMax color decoding with a 1x1 FLOAT Conv over one-hot channels to reduce INT64 memory. | promoted |
| threshold01-drop-period5 | rule_redesign | expected_delta 0.1-1.0: remove only period 5 while keeping known-needed periods 8 and 9; failure showed the local generator still needs p=5. | fails_local |

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
| exp011 | impl_opt | drop-fallback-period-ok | passes_local | 13.351346806118576 | 111492 | 3045 | 0.0224035890815 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | drop-unused-fallback-size | passes_local | 13.351355536959836 | 111492 | 3044 | 8.73084126063e-06 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | select-before-pad | passes_local | 13.412514522204688 | 104697 | 3044 | 0.0611589852449 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | cast-seen-counts | passes_local | 13.412523803765476 | 104697 | 3043 | 9.2815607875e-06 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | period-minmax-color-grid | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp016 | impl_opt | period-minmax-color-grid-opset13 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp017 | impl_opt | period-minmax-color-grid-opset13-v2 | passes_local | 14.26758951972832 | 41995 | 3822 | 0.855065715963 | promoted | Auto promoted after canonical re-score. |
| exp018 | impl_opt | full-argmax-minmax-color-grid | passes_local | 14.76401237516649 | 24076 | 3813 | 0.496422855438 | promoted | Auto promoted after canonical re-score. |
| exp019 | impl_opt | p7-tile-shared-end | passes_local | 14.780425150506924 | 24076 | 3359 | 0.0164127753404 | promoted | Auto promoted after canonical re-score. |
| exp021 | impl_opt | drop-unused-one-i64 | passes_local | 14.780461600961647 | 24076 | 3358 | 3.64504547239e-05 | promoted | Auto promoted after canonical re-score. |
| exp022 | impl_opt | conv-color-map | passes_local | 14.920712107660183 | 20476 | 3368 | 0.140250506699 | promoted | Auto promoted after canonical re-score. |
| exp023 | rule_redesign | hu-smoke-drop-period9 | fails_local | 0.0 | 17574 | 2598 | -14.9207121077 | fails_local | Candidate did not pass local validation. |
| exp024 | rule_redesign | hu-smoke-drop-period8 | fails_local | 0.0 | 17693 | 2666 | -14.9207121077 | fails_local | Candidate did not pass local validation. |
| exp025 | rule_redesign | threshold01-drop-period5 | fails_local | 0.0 | 18044 | 2822 | -14.9207121077 | fails_local | Candidate did not pass local validation. |

## Archived Summary
- None yet.
