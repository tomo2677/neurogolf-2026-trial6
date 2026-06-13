# task009 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 15.567476537804732 | 11523 | 965 | 2026-06-13T14:01:36+09:00 | exp069 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp047 | impl_opt | argmax-first-last-multi | fails_local | 0.0 | 16642 | 986 | -15.2136644323 | fails_local | Candidate did not pass local validation. |
| exp048 | impl_opt | colorized-reducemax-cell-select | passes_local | 15.235946943455449 | 16406 | 991 | 0.0222825111807 | promoted | Auto promoted after canonical re-score. |
| exp049 | impl_opt | actual-color-code-cell-select | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp050 | impl_opt | actual-color-code-gather | passes_local | 15.277015658451043 | 15698 | 999 | 0.0410687149956 | promoted | Auto promoted after canonical re-score. |
| exp051 | impl_opt | uint8-occ-line-fill | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp052 | impl_opt | uint8-occ-reducemax-multi | passes_local | 15.346449253673395 | 14578 | 999 | 0.0694335952224 | promoted | Auto promoted after canonical re-score. |
| exp053 | impl_opt | drop-unused-one1111 | passes_local | 15.346513452947965 | 14578 | 998 | 6.41992745702e-05 | promoted | Auto promoted after canonical re-score. |
| exp054 | impl_opt | vertical-fill-axis2 | passes_local | 15.397887847047656 | 13778 | 1018 | 0.0513743940997 | promoted | Auto promoted after canonical re-score. |
| exp055 | impl_opt | selected-end-delta | passes_local | 15.398902148787927 | 13762 | 1019 | 0.00101430174027 | promoted | Auto promoted after canonical re-score. |
| exp056 | impl_opt | slotwise-line-fill-max | passes_local | 15.427519644654026 | 13358 | 1006 | 0.0204658305923 | promoted | Auto promoted after canonical re-score. |
| manual057 | impl_opt | square-valid-transpose | passes_local | 15.435908955602331 | 13238 | 1006 | 0.00838931094831 | promoted | Direct canonical re-score; square grids allow `col_valid = Transpose(row_valid)`. |
| manual058 | impl_opt | line-fill-singletons | passes_local | 15.505684417503362 | 12278 | 1006 | 0.069775461901 | promoted | Direct canonical re-score; line-fill with `left == right` preserves singleton endpoints, removing `span_multi`, `occ_bool`, and `cell_fill_lines`. |
| manual059 | impl_opt | line-fill-leq | passes_local | 15.556644956914344 | 11638 | 986 | 0.050960539411 | promoted | Direct canonical re-score; use `LessOrEqual(left, idx)` and `LessOrEqual(idx, right)` to drop per-axis `right_end` tensors and plus-one index params. |
| exp057 | impl_opt | resize-sizes | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp058 | impl_opt | resize-sizes-opset13 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp059 | impl_opt | resize-sizes-opset13-split-input | passes_local | 15.556961863904796 | 11638 | 982 | 0.000316906990452 | promoted | Auto promoted after canonical re-score. |
| exp060 | impl_opt | split-equal-no-input | passes_local | 15.557278871357125 | 11638 | 978 | 0.000317007452329 | promoted | Auto promoted after canonical re-score. |
| exp061 | impl_opt | grid-mask-int32 | passes_local | 15.55799250576873 | 11638 | 969 | 0.000713634411605 | promoted | Auto promoted after canonical re-score. |
| exp062 | impl_opt | reuse-zero-score | passes_local | 15.558071829926945 | 11638 | 968 | 7.93241582144e-05 | promoted | Auto promoted after canonical re-score. |
| exp063 | impl_opt | grid-pixel-axes3 | passes_local | 15.558230497122835 | 11638 | 966 | 0.00015866719589 | promoted | Auto promoted after canonical re-score. |
| exp064 | impl_opt | sparse-body-mask | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp065 | impl_opt | float-occ-linefill | passes_local | 15.571006141982473 | 11478 | 966 | 0.0127756448596 | promoted | Auto promoted after canonical re-score. |
| exp066 | impl_opt | zero-f32-shared | passes_local | 15.566516076709608 | 11534 | 966 | 7.99968001708e-05 | promoted | Auto promoted after canonical re-score. |
| exp068 | impl_opt | topk10-mask-black-grid | passes_local | 15.567396464136914 | 11523 | 966 | 0.000880387427307 | promoted | Auto promoted after canonical re-score. |
| exp069 | impl_opt | cast-zero-f32 | passes_local | 15.567476537804732 | 11523 | 965 | 8.00736678173e-05 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- Official ERROR repair: the 2026-06-12 single-task official submission for the previous model returned `SubmissionStatus.ERROR` despite local `passes_local` and `passes_rules`. The common pattern with other ERROR tasks was `TopK(UINT8)`. Repaired by keeping `object_scores` as `FLOAT` through `TopK`; expected local score after repair is about `15.566436079909437`.
- Reached `15.571006141982473` by moving the solution to 10x10 cell space: exclude the grid color, keep only `TopK(k=4)` object colors, sample each object's cell occupancy with `Slice(steps=[1,3,3])`, use endpoint line-fill from `ArgMax(select_last_index)`, merge body/grid with `Where(body_mask, cell_or_black30_u8, grid_color_u8)`, use actual `0..9` color codes, select filled cell colors with per-slot candidates plus final `Max`, and avoid the 4-color `occ` concat.
- Latest reductions: square input grids allow `col_valid = Transpose(row_valid)`; singleton line-fill (`left == right`) already preserves original endpoint cells, so `span_multi`, `occ_bool`, and `cell_fill_lines` are unnecessary; `LessOrEqual` bounds remove per-axis `right_end` tensors and plus-one index params; exp059 moves to opset13 so `Resize` can use `sizes=[1,1,30,30]` instead of `roi+scales`; exp060 confirms equal `Split` works without a `split4` input; exp061 compares the grid color as `INT32` and reuses `input_channel_ids`, removing `channel_indices_i64`; exp062 reuses `zero_score_u8` for small `UINT8` thresholds; exp063 slices `grid_pixel` with existing `slice_axes3`; exp065 removes the four `occ -> UINT8` casts and runs line-fill directly on the `FLOAT` occupancy slices, saving 160 bytes.
- Task-local blocker: score 20 requires `memory_bytes_approx + params < 149`, but current total is `11478 + 966 = 12444`. The graph still materializes dense 30x30 tensors (`valid_area`, `cell_or_black30_u8`, `color30_raw_u8`, `color_grid_u8`), four selected-color `FLOAT [1,1,10,10]` occupancy slices, many 10x10 line-fill masks, and a 900-param static `body_mask`. Factorizing `body_mask` into row/column masks saves params but adds equivalent 30x30 intermediates; `ReduceSum(UINT8)` is checker-unsupported; and avoiding the remaining dense tensors would require a different sparse/direct output construction accepted by the scorer.
- Revisited after task010 `UINT8 Add` and task002 `Sub` wins: the expensive path is not a scalar rank/seed cascade but 10x10 line-fill with `ArgMax(INT64)` endpoints, many bool masks, and dense 30x30 output composition. Casting endpoint indices down would only add tensors because `ArgMax` still outputs `INT64`; `OneHot` remains unavailable locally.
- Revisited after task010/task002 `ReduceSum` height/count wins: task009 valid-area is already `row_valid + Transpose`, so there is no last-row `ArgMax` to remove. The remaining `ArgMax` nodes are line-fill endpoints; counts are insufficient because the fill needs left/right positions, not just object extent.
