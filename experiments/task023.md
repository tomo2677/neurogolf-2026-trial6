# task023 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 10.62027396320226 | 1753556 | 4512 | 2026-06-13T06:00:31+09:00 | exp006 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp002 | impl_opt | h002 | passes_local | 10.087275463881577 | 2988096 | 7712 | 0.685827600454 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | reduce-steps-10 | passes_local | 10.267913950709392 | 2494280 | 6432 | 0.180638486828 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | reduce-steps-9 | passes_local | 10.372153991401705 | 2247372 | 5792 | 0.104240040692 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | reduce-steps-8 | passes_local | 10.488538196543056 | 2000464 | 5152 | 0.116384205141 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | reduce-steps-7 | passes_local | 10.62027396320226 | 1753556 | 4512 | 0.131735766659 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
