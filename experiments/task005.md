# task005 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 14.246361518094261 | 46675 | 125 | 2026-06-12T07:28:06+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp030 | impl_opt | cast-flat-color | passes_local | 13.871619992885433 | 67232 | 844 | 0.00645716152787 | promoted | Auto promoted after canonical re-score. |
| exp031 | impl_opt | topk-flat-present | passes_local | 13.872134256361285 | 67200 | 841 | 0.000514263475852 | promoted | Auto promoted after canonical re-score. |
| exp032 | impl_opt | int32-dynamic-slice-indices | passes_local | 13.875077989076818 | 67000 | 841 | 0.00294373271553 | promoted | Auto promoted after canonical re-score. |
| exp033 | impl_opt | u8-topk-derive-end | passes_local | 13.875638279262807 | 66971 | 832 | 0.000560290185989 | promoted | Auto promoted after canonical re-score. |
| exp034 | impl_opt | selected-seed-slice-axes3 | passes_local | 13.87618412677837 | 66931 | 835 | 0.000545847515562 | promoted | Auto promoted after canonical re-score. |
| exp035 | rule_redesign | top3-selected-colors | fails_local | 0.0 | 60814 | 835 | -13.8761841268 | fails_local | Candidate did not pass local validation. |
| exp036 | impl_opt | opset10-pad-attrs | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp037 | impl_opt | opset10-pad-attrs-default-topk | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp038 | impl_opt | opset10-pad-attrs-float-topk | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp039 | rule_redesign | seed-selected0-current | fails_local | 0.0 | 60449 | 835 | -13.8761841268 | fails_local | Candidate did not pass local validation. |
| exp040 | impl_opt | topidx-channelid-slice-delta | passes_local | 13.876700743393918 | 66903 | 828 | 0.000516616615549 | promoted | Auto promoted after canonical re-score. |
| exp041 | rule_redesign | seed-reducesum-score | fails_local | 0.0 | 64043 | 835 | -13.8767007434 | fails_local | Candidate did not pass local validation. |
| exp042 | rule_redesign | seed-reducesum-color-tiebreak | fails_local | 0.0 | 64059 | 836 | -13.8767007434 | fails_local | Candidate did not pass local validation. |
| exp043 | rule_redesign | seed-rowcol-scalar-score | passes_local | 13.913681566891364 | 64435 | 837 | 0.0369808234974 | promoted | Auto promoted after canonical re-score. |
| exp044 | rule_redesign | seed-row-score | passes_local | 13.916504520591769 | 64251 | 837 | 0.0028229537004 | promoted | Auto promoted after canonical re-score. |
| exp045 | impl_opt | dilated-repeat-kernel | passes_local | 13.92651016270909 | 64251 | 189 | 0.0100056421173 | promoted | Auto promoted after canonical re-score. |
| exp046 | impl_opt | color-before-repeat-matmul | passes_local | 13.981764376886359 | 60787 | 189 | 0.0552542141773 | promoted | Auto promoted after canonical re-score. |
| exp047 | impl_opt | float-internal-no-casts | passes_local | 13.46689942809314 | 101849 | 189 | -0.514864948793 | not_better | Passed but did not improve local_points. |
| exp048 | impl_opt | drop-seed-color-tiebreak | fails_local | 0.0 | 60771 | 188 | -13.9817643769 | fails_local | Candidate did not pass local validation. |
| exp049 | rule_redesign | repeat-mask-direction-selector | passes_local | 14.244994930932485 | 46675 | 189 | 0.263230554046 | promoted | Auto promoted after canonical re-score. |
| exp050 | rule_redesign | seed-maxrow-color-score | fails_local | 0.0 | 46659 | 189 | -14.2449949309 | fails_local | Candidate did not pass local validation. |
| exp051 | impl_opt | drop-unused-shift-pads | passes_local | 14.246361518094261 | 46675 | 125 | 0.00136658716178 | promoted | Auto promoted after canonical re-score. |
| exp052 | impl_opt | float-repeat-path | passes_local | 13.69193622266597 | 81349 | 127 | -0.554425295428 | not_better | Passed but did not improve local_points. |
| exp053 | rule_redesign | argmax-gather-repeat | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp054 | rule_redesign | argmax-gather-repeat-opset11 | fails_local | 0.0 | 53707 | 125 | -14.2463615181 | fails_local | Candidate did not pass local validation. |

## Archived Summary
- Reached `14.246361518094261` / `46675` bytes / `125` params. The current pipeline uses local-data-bounded `TopK(k=4)`, dynamic 3-axis `Slice` for selected channels, a scalar seed score `cell_count + 0.1 * max_row_count + 0.01 * color_id`, a dynamic seed slice, 8-direction repeated seed masks via dilated `Conv`, fragment direction detection by overlapping selected fragments with each direction's repeat mask, and a shared `repeat_matrix` consumed by per-slot `MatMul`.
- Recent improvements: exp043 replaced `AveragePool(3x3)` seed scoring with cheap row-count scalar scoring; exp044 found `max_row_count` alone was sufficient and removed `col_counts`; exp045 replaced dense 13x13 repeat kernels with 4x4/4x1/1x4 dilated kernels, dropping params from `837` to `189`; exp046 moved color multiplication before `MatMul`; exp049 reused the existing repeat masks for direction selection and removed the 8 one-step seed shifts plus neighbor flats; exp051 removed now-unused shift-pad initializers.
- Confirmed non-viable shortcuts: `TopK(k=3)` fails local validation (`168/266` passed), fixed `selected_0` as seed fails (`87/266` passed), pure `ReduceSum` seed scoring fails 2 generated examples, color-id tie-break still fails 1 generated example, dropping seed color tie-break fails, `FLOAT` internal tensors increase memory due larger selected/repeat tensors, `opset10` Pad attributes fail because this graph needs `UINT8` final `Pad` and modern `TopK` behavior, and sparse Conv kernels are rejected by ONNX shape inference (`Conv` weight does not support sparse tensor).
- Current task-local blocker: score 20 requires `memory_bytes_approx + params < 149`, but `repeat_matrix` alone is `7056` bytes and the graph still needs dense 21x21/441-element selected slices, seed slice, repeat flats, selected flats, colored slot outputs, and `color30_u8` (`900` bytes). Local data has 98 examples with 4 nonzero colors, so `k=4` is required and the prior `k=3` shortcut is structurally invalid. A second-pass `ArgMax + Gather(repeat_matrix)` replacement fails local validation (`210/266` passed) because some colors legitimately need multiple repeat directions; local direction analysis found 56 multi-direction color cases. The current `Gemm`/`MatMul` design is already the cheaper passing alternative to per-slot/per-direction dense fragment grids; further large gains require sparse or dynamic per-stamp placement that the scorer's static-shape memory model and supported ONNX operators do not provide.
- Revisited after task002 `MaxPool(UINT8)` line-of-sight wins: not applicable to the repeated-stamp core. The repeat masks are spaced 4 pixels apart and depend on the selected seed stamp, so the current dilated `Conv` plus shared `repeat_matrix` remains the cheapest passing representation found; replacing it with per-direction dense selected outputs would reintroduce larger tensors.
- Revisited after task010/task002 `UINT8 Add/Sub` wins: not applicable to the dominant repeated-stamp matrix path. `Conv` supports only float-like tensors in the local ONNX schema, and `Gemm/MatMul` do not support `UINT8`; moving selected/repeat tensors away from `FLOAT16` would either fail checker/ORT or increase memory with `INT32`. The main blocker remains dense dynamic 21x21 selected masks plus the 8x441 repeat matrix.
- Revisited after task010/task002 `ReduceSum` height/count wins: task005 already uses `ReduceSum` for seed scoring where it helps. The remaining large costs are selected 21x21 slices, repeat `Conv` outputs/flats, and the shared `repeat_matrix`; there is no valid-area `ArgMax` or scalar height-rank path left to replace.
- Revisited after task004 selected-slot audit: avoiding the extra seed re-slice by dynamically choosing from `selected_0..3` would require either concatenating the four selected masks and gathering one dense 21x21 plane, or multiplying each selected mask by a scalar selector before merging. Both add more dense `FLOAT16` tensors than the current seed re-slice, so the existing seed path remains cheaper.
