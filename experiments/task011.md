# task011 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.235703993549482 | 2289 | 66 | 2026-06-13T10:27:26+09:00 | exp012 |

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

## Archived Summary
- None yet.
