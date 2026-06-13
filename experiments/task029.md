# task029 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 13.393183543615244 | 109698 | 146 | 2026-06-13T14:21:52+09:00 | exp014 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | passes_local | 12.140805610766446 | 382403 | 1903 | 0.358293696231 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | int32-frame-indices | passes_local | 12.189474784925212 | 364147 | 1903 | 0.0486691741588 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | crop23-window | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp004 | impl_opt | crop23-window-split-grids | passes_local | 12.227381350338574 | 349465 | 2969 | 0.0379065654134 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | reuse-k1 | passes_local | 12.227384187753293 | 349465 | 2968 | 2.83741471918e-06 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | f16-shape-counts | passes_local | 12.323926850228974 | 317031 | 2968 | 0.0965426624757 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | color-grid-masks | passes_local | 12.400683286658618 | 293451 | 2905 | 0.0767564364296 | promoted | Auto promoted after canonical re-score. |
| exp008 | impl_opt | gather-1d-inner-crop | passes_local | 12.409679073166743 | 290806 | 2896 | 0.00899578650812 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | broadcast-frame-grids | passes_local | 13.115855865650198 | 144806 | 144 | 0.706176792483 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | int32-inner-gather | passes_local | 13.145486833310368 | 140574 | 144 | 0.0296309676602 | promoted | Auto promoted after canonical re-score. |
| exp011 | rule_redesign | size25-internal-crop | passes_local | 13.245509702195179 | 127184 | 140 | 0.100022868885 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | axis-gather-inner-crop | passes_local | 13.311634938024577 | 119038 | 139 | 0.0661252358294 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | direct-zero-f16 | passes_local | 13.311651719926974 | 119036 | 139 | 1.67819023975e-05 | promoted | Auto promoted after canonical re-score. |
| exp014 | rule_redesign | top5-frame-colors | passes_local | 13.393183543615244 | 109698 | 146 | 0.0815318236883 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
