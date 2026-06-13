# task035 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.485209693200073 | 4864 | 124 | 2026-06-13T12:39:01+09:00 | exp011 |

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
| exp006 | impl_opt | reuse-color-bounds | passes_local | 16.240174404685703 | 6240 | 133 | 0.00125450855405 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | pad18-axes | passes_local | 16.243632440197025 | 6240 | 111 | 0.00345803551132 | promoted | Auto promoted after canonical re-score. |
| exp008 | impl_opt | broadcast-edge-colors | passes_local | 16.338360204218734 | 5670 | 107 | 0.0947277640217 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | rowcol-edge-masks | passes_local | 16.452083635940923 | 5068 | 88 | 0.113723431722 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | orig-border-concat | passes_local | 16.48280680858376 | 4864 | 136 | 0.0307231726428 | promoted | Auto promoted after canonical re-score. |
| exp011 | impl_opt | edge-slice-hw-axes | passes_local | 16.485209693200073 | 4864 | 124 | 0.00240288461631 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
