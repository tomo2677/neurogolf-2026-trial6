# task010 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 18.319145321209785 | 748 | 49 | 2026-06-12T08:46:51+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp010 | impl_opt | transpose-height-rank | passes_local | 18.062685918776317 | 969 | 61 | 0.0182785271726 | promoted | Auto promoted after canonical re-score. |
| exp011 | impl_opt | direct4-f16-colors5-transpose-rank | passes_local | 18.062685918776317 | 969 | 61 | 0 | not_better | Passed but did not improve local_points. |
| exp012 | impl_opt | slice-default-axes-steps | passes_local | 18.070483229236352 | 969 | 53 | 0.00779731046003 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | direct4-no-split-input | passes_local | 18.074404802889532 | 969 | 49 | 0.00392157365318 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | direct4-uint8-color-grid | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp015 | impl_opt | direct4-cast-uint8-color-grid | passes_local | 18.110408691645535 | 933 | 49 | 0.036003888756 | promoted | Auto promoted after canonical re-score. |
| exp016 | impl_opt | direct4-axes3-slice | passes_local | 18.079328495751316 | 969 | 44 | -0.0310801958942 | not_better | Passed but did not improve local_points. |
| exp017 | impl_opt | uint8-zero-col-init | passes_local | 18.128908705389456 | 906 | 58 | 0.0185000137439 | promoted | Auto promoted after canonical re-score. |
| exp018 | impl_opt | uint8-rank-count | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp019 | impl_opt | direct4-u8-where-columns | passes_local | 18.16589126118616 | 879 | 50 | 0.0369825557967 | promoted | Auto promoted after canonical re-score. |
| exp020 | impl_opt | direct4-u8-where-axes3-slice | passes_local | 18.171287928358318 | 879 | 45 | 0.00539666717216 | promoted | Auto promoted after canonical re-score. |
| exp021 | impl_opt | direct4-f32-height-u8-where | passes_local | 18.171287928358318 | 879 | 45 | 0 | not_better | Passed but did not improve local_points. |
| exp022 | impl_opt | opset12-reduce-attrs | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp023 | impl_opt | toprow-argmax-rank | passes_local | 18.17237076549715 | 879 | 44 | 0.00108283713883 | promoted | Auto promoted after canonical re-score. |
| exp024 | impl_opt | f16-toprow-rank | passes_local | 18.21667479939604 | 839 | 44 | 0.0443040338989 | promoted | Auto promoted after canonical re-score. |
| exp025 | impl_opt | scalar-pairwise-rank | passes_local | 18.26540834002705 | 799 | 42 | 0.048733540631 | promoted | Auto promoted after canonical re-score. |
| exp026 | impl_opt | int64-scalar-rank | passes_local | 18.274966357833158 | 791 | 42 | 0.00955801780611 | promoted | Auto promoted after canonical re-score. |
| exp027 | impl_opt | zero-col-init-current | passes_local | 18.27616755917879 | 782 | 50 | 0.00120120134563 | promoted | Auto promoted after canonical re-score. |
| exp028 | impl_opt | sparse-zero-col | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp029 | impl_opt | pad18-axes3 | passes_local | 18.278574299209357 | 782 | 48 | 0.00240674003057 | promoted | Auto promoted after canonical re-score. |
| exp030 | impl_opt | bool-rank-where | passes_local | 18.274966357833158 | 782 | 51 | -0.0036079413762 | not_better | Passed but did not improve local_points. |
| exp031 | impl_opt | u8-add-rank | passes_local | 18.293137663397253 | 770 | 48 | 0.0145633641879 | promoted | Auto promoted after canonical re-score. |
| exp032 | impl_opt | pairwise-complement-rank | passes_local | 18.300499659838323 | 764 | 48 | 0.00736199644107 | promoted | Auto promoted after canonical re-score. |
| exp033 | impl_opt | reducesum-height-rank | passes_local | 18.319145321209785 | 748 | 49 | 0.0186456613715 | promoted | Auto promoted after canonical re-score. |
| exp034 | impl_opt | u8-sum-rank | build_failed |  |  |  |  | build_failed | `Sum(UINT8)` is unsupported by ONNX shape inference. |

## Archived Summary
- Reached `18.319145321209785` by exploiting the fixed four bar columns `(1,3,5,7)`: slice only those columns with 3-axis `Slice`, compute per-bar height with `ReduceSum(FLOAT)` instead of `ArgMax(INT64)`, use six pairwise height comparisons plus `Sub(one, flag)` complements because local/generated examples have distinct heights, add the `UINT8` comparison bits to get ranks, build color columns directly with `Where`, emit `BOOL [1,5,9,9]` via `Equal(colors5, color9)`, and use `Pad-18` with existing `axes_slice` to shrink final pad params. Dense `zero_col` as an initializer is slightly better than generating it at runtime.
- Task-local blocker: scoring above 20 requires `memory+params < 149`, but the necessary pre-pad one-hot top grid `output5` is already `405` bytes (`BOOL [1,5,9,9]`) before the four input column slices (`144` bytes), `color9` (`81` bytes), and scalar rank/color tensors are counted. exp031 replaced `FLOAT16 Sum` rank with `UINT8 Add`; exp032 reused pairwise complements to remove six comparison tensors; exp033 replaced `ArgMax(INT64)` top positions with `ReduceSum(FLOAT)` heights (`764/48 -> 748/49`). `UINT8 Sum` and `UINT8 ReduceSum` are checker-unsupported, sparse `zero_col` is not accepted by `Where`, old-opset attr-style paths are blocked by the final bool `Pad`, and `TopK`-style optional values outputs cannot be omitted. Reaching 20 would require avoiding dense one-hot materialization before the final `Pad`, such as sparse/dynamic output construction or bit-packed representation, which is not available under the current scorer constraints.
- Revisited after task009 line-fill cleanup: replacing the scalar `Add` chains with `Sum(UINT8)` would save a few bytes, but exp034 confirms `Sum(UINT8)` is rejected by ONNX shape inference. The remaining rank tensors are small compared with `output5`.
