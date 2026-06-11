# task008 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.41160304957743 | 5276 | 93 | 2026-06-12T07:00:05+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp026 | impl_opt | where-select-shift | score_failed |  |  |  |  | score_failed | Candidate did not pass local validation. |
| exp027 | impl_opt | share-base-indices | passes_local | 15.895909427866528 | 8914 | 78 | 0.001777778246 | promoted | Auto promoted after canonical re-score. |
| exp028 | impl_opt | int32-offset-arithmetic | passes_local | 15.89679950358172 | 8906 | 78 | 0.000890075715192 | promoted | Auto promoted after canonical re-score. |
| exp029 | impl_opt | pad18-specific-axes | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp030 | impl_opt | pad18-specific-axes-reducemax-input | passes_local | 15.898136104359901 | 8906 | 66 | 0.00133660077818 | promoted | Auto promoted after canonical re-score. |
| exp031 | rule_redesign | constant-valid16-no-black-slice | fails_local | 0.0 | 7626 | 60 | -15.8981361044 | fails_local | Candidate did not pass local validation. |
| exp032 | impl_opt | int32-static-slice-indices | passes_local | 15.898136104359901 | 8906 | 66 | 0 | not_better | Passed but did not improve local_points. |
| exp033 | impl_opt | range-channel-ids | passes_local | 15.887382716552432 | 9006 | 63 | -0.0107533878075 | not_better | Passed but did not improve local_points. |
| exp034 | impl_opt | select-offset-direct | passes_local | 15.900032685505188 | 8890 | 65 | 0.00189658114529 | promoted | Auto promoted after canonical re-score. |
| exp035 | impl_opt | bool-bbox-reducemax | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp036 | impl_opt | bbox-valid-rectangle | passes_local | 15.932722010865655 | 8570 | 97 | 0.0326893253605 | promoted | Auto promoted after canonical re-score. |
| exp037 | impl_opt | base-valid-coords | passes_local | 15.93352999844352 | 8594 | 66 | 0.000807987577865 | promoted | Auto promoted after canonical re-score. |
| exp038 | impl_opt | black-only-valid-bounds | passes_local | 15.93445421292705 | 8586 | 66 | 0.000924214483531 | promoted | Auto promoted after canonical re-score. |
| exp039 | impl_opt | dynamic-slice-shift | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp040 | impl_opt | dynamic-slice-shift-i32axes | passes_local | 15.946196485844045 | 8482 | 69 | 0.011742272917 | promoted | Auto promoted after canonical re-score. |
| exp041 | impl_opt | slice-end-1d | passes_local | 15.947132486848382 | 8474 | 69 | 0.000936001004337 | promoted | Auto promoted after canonical re-score. |
| exp042 | impl_opt | direct-valid-coords-after-slice | passes_local | 15.94806936477146 | 8450 | 85 | 0.000936877923078 | promoted | Auto promoted after canonical re-score. |
| exp043 | impl_opt | single-2d-shift-slice | passes_local | 16.06924126444173 | 7474 | 87 | 0.12117189967 | promoted | Auto promoted after canonical re-score. |
| exp044 | impl_opt | red-5x5-crop-pad | passes_local | 16.20669137250345 | 6507 | 83 | 0.137450108062 | promoted | Auto promoted after canonical re-score. |
| exp045 | impl_opt | cast-red-crop-only | passes_local | 16.23064829220103 | 6351 | 83 | 0.0239569196976 | promoted | Auto promoted after canonical re-score. |
| exp046 | impl_opt | precolor-red-crop-pad | passes_local | 16.267211675026882 | 6120 | 83 | 0.0365633828259 | promoted | Auto promoted after canonical re-score. |
| exp047 | rule_redesign | black-edge-probes | passes_local | 16.38658795084322 | 5416 | 89 | 0.119376275816 | promoted | Auto promoted after canonical re-score. |
| exp048 | impl_opt | blue-first-plus-one | passes_local | 16.38949863145011 | 5400 | 89 | 0.00291068060689 | promoted | Auto promoted after canonical re-score. |
| exp049 | rule_redesign | blue-template-pad | passes_local | 16.41160304957743 | 5276 | 93 | 0.0221044181273 | promoted | Auto promoted after canonical re-score. |
| exp050 | impl_opt | int32-dynamic-pad | build_failed |  |  |  |  | build_failed | Candidate did not build. |

## Archived Summary
- Reached `16.41160304957743` by replacing the original 30x30 shift enumeration with a 16x16 crop, compact bbox via `ArgMax`, scalar `UINT8` color-grid output plus final graph-output `Equal`, dynamic 5x5 red crop padding, black edge probes for actual-grid bounds, and a blue 2x2 template dynamically padded from the blue bbox.
- Key later improvements: exp047 replaced the full 16x16 black slice with bottom/right black probes (`cols 2:5` for bottom, `rows 0:2` for right); exp048 used the fixed 2x2 blue size so `blue_bottom/right = blue_top/left + 1`; exp049 reduced the blue bbox slice to a `14x15` probe and generated the unchanged blue square from a padded `UINT8 [1,1,2,2]` template.
- Current task-local blocker: score 20 requires `memory_bytes_approx + params < 149`, but current total is `5276 + 93 = 5369`. The current graph still has dense `red` (`FLOAT [1,1,16,16]`, 1024 bytes), `blue` probe (`FLOAT [1,1,14,15]`, 840 bytes), `color_grid_u8` (`UINT8 [1,1,30,30]`, 900 bytes), and multiple 16x16 composition tensors (`red_color_u8`, `blue_color_u8`, `color_nonblack_u8`, `valid_bool`, `color16_u8`). The black probe ranges are minimal on the local data (`bottom` needs width 3; `right` needs height 2), blue top/left ranges are exactly `top=0..13` and `left=0..14`, and red bbox reaches `bottom=15` and `right=15`, so the current blue probe and 16x16 red slice cannot be substantially narrowed without losing examples. The scalar `color_grid_u8` remains cheaper than a pre-pad dense 10-channel one-hot tile, and fixed red `2` / blue `8` channel positions prevent a compact 3-channel pad like task003/task006. exp050 confirmed dynamic `Pad` pads must be `INT64`, so the current `INT32 -> INT64` casts for red/blue pads are required. Further movement toward 20 would require a fundamentally different sparse/scorer-compatible output construction or avoiding dense red/blue spatial slices.
- Revisited after task002 `MaxPool(UINT8)` line-of-sight wins: not applicable. task008 needs object bbox from dense red/blue spatial regions, not directional visibility through a wall; the large tensors are the object slices and output composition.
- Revisited after task010 `UINT8 Add`, task002 `Sub`, and task001 `OneHot` checks: the dominant tensors are red/blue spatial slices, dynamic pad composition, and the scalar `color_grid_u8`; there is no large `Equal(...)->Cast(UINT8)` seed pair to collapse. `OneHot` is not implemented locally, and direct compact channel pads cannot represent fixed red `2` plus blue `8` without a dense color grid or full one-hot construction.
- Revisited after task004/task007 dense-output audits: replacing the scalar `color_grid_u8` with a channel tile would be larger because the fixed colors occupy channels 2 and 8, forcing at least 9 channels. Red bbox/crop also still needs a 16x16 red channel slice; splitting it into probes would not recover the arbitrary red shape needed for the shifted 5x5 crop.
