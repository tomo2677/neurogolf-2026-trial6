# task035 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.238919896131655 | 6240 | 141 | 2026-06-13T05:23:39+09:00 | exp005 |

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
| exp004 | impl_opt | base-orig-max | passes_local | 16.208058011543883 | 6440 | 141 | 0.101081505939 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | fixed-top-right-masks | passes_local | 16.238919896131655 | 6240 | 141 | 0.0308618845878 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
