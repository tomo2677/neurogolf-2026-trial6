# task018 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 8.989728341011567 | 8975056 | 2800 | 2026-06-13T06:05:00+09:00 | exp005 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | passes_local | 8.298108748965113 | 17925504 | 2800 |  | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | h001 | passes_local | 8.298108748965113 | 17925504 | 2800 | 0 | not_better | Passed but did not improve local_points. |
| exp003 | impl_opt | h002 | passes_local | 8.946564904291776 | 9371056 | 2800 | 0.648456155327 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | grow-steps-15 | passes_local | 8.975791382000043 | 9101056 | 2800 | 0.0292264777083 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | grow-steps-8 | passes_local | 8.989728341011567 | 8975056 | 2800 | 0.0139369590115 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
