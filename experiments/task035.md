# task035 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.10697650560529 | 7140 | 141 | 2026-06-13T03:56:12+09:00 | exp003 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | rule_redesign | edge-only-color-grid | passes_local | 16.06897678414397 | 7430 | 133 | 0.197890157424 | promoted | Auto promoted after canonical re-score. |
| exp002 | rule_redesign | fixed-right-col5 | passes_local | 16.106701855782084 | 7140 | 143 | 0.0377250716381 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | fixed-right-col5-no-unused | passes_local | 16.10697650560529 | 7140 | 141 | 0.000274649823204 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
