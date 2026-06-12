# task033 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.377825405182378 | 1877 | 166 | 2026-06-13T06:52:41+09:00 | exp005 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | pad-fill30-bool | passes_local | 16.840339262936624 | 3383 | 114 | 0.276105455657 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | cast-bool-masks | passes_local | 16.840625263224574 | 3383 | 113 | 0.00028600028795 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | central-bg-template-reuse | passes_local | 17.139429214461337 | 2425 | 168 | 0.298803951237 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | pad-axes | passes_local | 17.14020081943789 | 2425 | 166 | 0.000771604976553 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | block-fill-concat | passes_local | 17.377825405182378 | 1877 | 166 | 0.237624585744 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
