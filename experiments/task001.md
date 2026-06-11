# task001 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.970027088293612 | 1095 | 35 | 2026-06-12T08:50:28+09:00 | exp033 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp009 | impl_opt | broadcast-spatial | passes_local | 16.097136327803778 | 7348 | 5 | 0.0336975324972 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | reshape-attr-opset4 | score_failed |  |  |  |  | score_failed | Candidate did not pass local validation. |
| exp011 | impl_opt | bool-mask | passes_local | 16.118302593530696 | 7195 | 4 | 0.0211662657269 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | bool-output9 | passes_local | 16.451889705949043 | 5153 | 4 | 0.333587112418 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | uint8-output-pad | passes_local | 17.079916800946766 | 2723 | 29 | 0.628027094998 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | split-color-selector | passes_local | 17.082463646056368 | 2724 | 21 | 0.0025468451096 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | bool-padded-output | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp016 | impl_opt | bool-padded-output-split-input | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp017 | impl_opt | bool-padded-output-opset13 | passes_local | 17.42904141683099 | 1914 | 27 | 0.346577770775 | promoted | Auto promoted after canonical re-score. |
| exp018 | impl_opt | pad-default-false | passes_local | 17.429556747942627 | 1914 | 26 | 0.000515331111636 | promoted | Auto promoted after canonical re-score. |
| exp019 | impl_opt | equal-mask | passes_local | 17.433688985227537 | 1905 | 27 | 0.00413223728491 | promoted | Auto promoted after canonical re-score. |
| exp020 | impl_opt | split-empty-output | score_failed |  |  |  |  | score_failed | Candidate did not pass local validation. |
| exp021 | impl_opt | flat-outer-spatial | fails_local | 0.0 | 1833 | 24 | -17.4336889852 | fails_local | Candidate did not pass local validation. |
| exp022 | impl_opt | color-grid-where | passes_local | 17.843043635384362 | 1247 | 36 | 0.409354650157 | promoted | Auto promoted after canonical re-score. |
| exp023 | impl_opt | color-grid-where-uint8 | passes_local | 17.909090177920017 | 1165 | 36 | 0.0660465425357 | promoted | Auto promoted after canonical re-score. |
| exp024 | impl_opt | depthtospace-mask | score_failed |  |  |  |  | score_failed | Candidate did not pass local validation. |
| exp025 | impl_opt | dynamic-onehot-and-spatial | fails_local | 0.0 | 1094 | 35 | -17.9090901779 | fails_local | Candidate did not pass local validation. |
| exp026 | impl_opt | direct-u8-dynamic-onehot | passes_local | 17.953352722151244 | 1104 | 45 | 0.0442625442312 | promoted | Auto promoted after canonical re-score. |
| exp027 | impl_opt | opset12-unsqueeze-attrs | passes_local | 17.95684008401166 | 1104 | 41 | 0.00348736186042 | promoted | Auto promoted after canonical re-score. |
| exp028 | impl_opt | int64-color-onehot | passes_local | 17.957713828060257 | 1103 | 41 | 0.000873744048597 | promoted | Auto promoted after canonical re-score. |
| exp029 | impl_opt | onehot-color | score_failed |  |  |  |  | score_failed | Candidate did not pass local validation. |
| exp030 | impl_opt | bool-pad18-color | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp031 | impl_opt | bool-pad18-color-reduceaxes | score_failed |  |  |  |  | score_failed | Candidate did not pass local validation. |
| exp032 | impl_opt | sub-black-color-vector | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp033 | impl_opt | sub-black-color-vector-opset14 | passes_local | 17.970027088293612 | 1095 | 35 | 0.0123132602334 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- Reached `17.970027088293612` by replacing the previous color-grid output with a direct `UINT8` one-hot top-left grid: build the 9x9 stamp mask from the zero channel, derive the active color vector with `ReduceMax(input) -> Cast(UINT8) -> Sub(black10)` using the local/generated invariant that every input has both black and one active nonzero color, and use `Where(spatial_mask, color_onehot, black10)` before final padding. exp033 moved to opset14 so `Sub(UINT8)` is supported; this adds `Unsqueeze` axes inputs but removes `ArgMax`, `Equal(colors10, color_id)`, and the `colors10` initializer.
- Task-local blocker: scoring above 20 requires `memory+params < 149`, but the dynamic color can be any channel `1..9`, so the necessary dense pre-pad one-hot tensor `output9_u8` is already `810` bytes (`UINT8 [1,10,9,9]`). A direct full-size color-grid-to-output alternative requires a `UINT8 [1,1,30,30]` intermediate (`900` bytes), and the older `color9 -> Equal` path is larger. A channel-block mask with `DepthToSpace` would shave only small mask overhead and is not executable in the local ORT profile path (`DepthToSpace(13)` not implemented). The remaining mask expansion (`spatial_bool6`/`spatial_bool9`) is also smaller than known `Resize`/runtime generation alternatives. Further major gains would require avoiding dense one-hot materialization with sparse/dynamic output construction or bit-packing, which is not available under the current scorer constraints.
- Revisited after task010 scalar-rank improvements: those tricks do not apply because task001 has no rank path; memory remains dominated by `output_bool9`, `spatial_bool6/spatial_bool9`, and `color9`.
- Revisited after `Pad-18` axes improvements from later tasks: not useful here because moving to opset18 would force `Unsqueeze` axes to become input tensors, offsetting the small final-pad param reduction.
- Revisited after task007-009 fixed sampling / resize ideas: fixed color sampling cannot beat the existing `ReduceMax(input)->ArgMax` path because color detection is only 40 bytes, while the dominant cost is `output9_u8` at 810 bytes. Replacing `Unsqueeze+And+Reshape` mask expansion with `Resize(nearest)` would save about 99 bytes, but `Resize(BOOL)` builds and then fails in ORT with `NOT_IMPLEMENTED`.
- Revisited after task003/task006 permuted `Equal([0,2,1], mask)` improvements: not applicable because task001 has a dynamic output color `1..9` and must also set the black channel outside the stamp. A simple `And(spatial, color_onehot)` leaves no active black channel, while scalar color-grid plus final `Equal` was already measured larger.
- Revisited after task009/task010 param-shaving wins: equal `Split`, int32 channel-id reuse, scalar zero reuse, and boolean rank rewrites do not touch task001's dominant tensors. The limiting cost remains the dynamic `UINT8 [1,10,9,9]` one-hot plus black-channel fill before final `Pad`.
- Revisited after task010 `UINT8 Add` improvement: dynamic `OneHot(axis=1)` would remove the intermediate bool color one-hot and shrink params, but `OneHot(11)` is not implemented in the local ORT path. A `Pad-18` bool-output variant would remove the `Cast(UINT8)`, but ORT rejects bool-payload `Where(16)`. The blocker remains dense dynamic one-hot materialization.
- Revisited after task010 `ReduceSum(FLOAT)` height improvement: no analogous rank/index scalar remains after exp033; task001 is now dominated by `output9_u8`, `spatial_bool6/spatial_bool9`, and the unavoidable input zero-channel slice.
