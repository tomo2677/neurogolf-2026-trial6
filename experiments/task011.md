# task011 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.44881313270385 | 1808 | 95 | 2026-06-13T19:10:58+09:00 | exp020 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | dynamic-gathernd-selected-block | passes_local | 16.519885816825184 | 4696 | 122 | 0.21338249271 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | dynamic-slice-selected-block | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp003 | impl_opt | dynamic-slice-selected-block | passes_local | 17.1195736557076 | 2536 | 109 | 0.599687838882 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | maxpool-block-select-fixed | passes_local | 17.151846913800476 | 2466 | 95 | 0.0322732580929 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | remove-unused-zero | passes_local | 17.152237462526394 | 2466 | 94 | 0.000390548725917 | promoted | Auto promoted after canonical re-score. |
| exp007 | rule_redesign | selected-channels-0-6 | passes_local | 17.1953407029439 | 2358 | 94 | 0.0431032404175 | promoted | Auto promoted after canonical re-score. |
| exp008 | impl_opt | seven-channel-bool-pad | passes_local | 17.218861490154985 | 2305 | 90 | 0.0235207872111 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | gather-expand-separators | passes_local | 17.226405532639806 | 2273 | 104 | 0.00754404248482 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | flatten-no8 | passes_local | 17.22724728353126 | 2273 | 102 | 0.000841750891453 | promoted | Auto promoted after canonical re-score. |
| exp011 | impl_opt | pad-axes-opset18 | passes_local | 17.228089743564237 | 2273 | 100 | 0.000842460032977 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | arithmetic-block-bounds | passes_local | 17.235703993549482 | 2289 | 66 | 0.00761424998525 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | int32-slice-bounds | passes_local | 17.25456438972562 | 2245 | 66 | 0.0188603961761 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | precomputed-sep-mask | passes_local | 17.264129680047432 | 2124 | 165 | 0.00956529032181 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | u8-no8 | passes_local | 17.267630777715613 | 2115 | 166 | 0.00350109766818 | promoted | Auto promoted after canonical re-score. |
| exp016 | impl_opt | bool7-pad | passes_local | 17.29158933274263 | 2062 | 165 | 0.023958555027 | promoted | Auto promoted after canonical re-score. |
| exp017 | impl_opt | channel8-slice-axes | passes_local | 17.29248780539966 | 2062 | 163 | 0.000898472657028 | promoted | Auto promoted after canonical re-score. |
| exp018 | impl_opt | argmin-has8-u8 | passes_local | 17.296992317520765 | 2053 | 162 | 0.00450451212111 | promoted | Auto promoted after canonical re-score. |
| exp019 | impl_opt | pad5-gather-separators | passes_local | 17.39610203147812 | 1959 | 47 | 0.0991097139574 | promoted | Auto promoted after canonical re-score. |
| exp020 | impl_opt | block-slices-has8 | passes_local | 17.44881313270385 | 1808 | 95 | 0.0527111012257 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
