# task007 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 18.01250975299901 | 1011 | 72 | 2026-06-13T03:35:42+09:00 | exp062 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp038 | impl_opt | pad18-axes3 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp039 | impl_opt | pad18-axes3-split-num-outputs | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp040 | impl_opt | pad18-axes3-int64 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp041 | impl_opt | topidx-channelid-slice-delta | passes_local | 17.718614336429717 | 1323 | 130 | 0.0129916357185 | promoted | Auto promoted after canonical re-score. |
| exp042 | rule_redesign | lshape-sampling | passes_local | 17.735269822070133 | 1309 | 120 | 0.0166554856404 | promoted | Auto promoted after canonical re-score. |
| exp043 | rule_redesign | fixed-points-max | passes_local | 17.836827609153357 | 1141 | 150 | 0.101557787083 | promoted | Auto promoted after canonical re-score. |
| exp044 | rule_redesign | two-residue-remaining | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp045 | rule_redesign | two-residue-remaining-axes-attr | passes_local | 17.953352722151244 | 1021 | 128 | 0.116525112998 | promoted | Auto promoted after canonical re-score. |
| exp046 | rule_redesign | two-residue-topk-u8 | passes_local | 17.976241045261556 | 995 | 128 | 0.0228883231103 | promoted | Auto promoted after canonical re-score. |
| exp047 | impl_opt | topk-drop-values | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp048 | impl_opt | no-known-squeeze | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp049 | impl_opt | rem0-keepdims | passes_local | 17.977131913917358 | 994 | 128 | 0.000890868655802 | promoted | Auto promoted after canonical re-score. |
| exp050 | impl_opt | rem0-sum-sub-u8 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp051 | impl_opt | cast-before-known-squeeze | passes_local | 17.98968813269277 | 980 | 128 | 0.0125562187754 | promoted | Auto promoted after canonical re-score. |
| exp052 | impl_opt | pad18-final-axes | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp053 | impl_opt | pad18-final-axes-reducemax-input | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp054 | impl_opt | pad18-final-axes-i64 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp055 | impl_opt | pad18-axes3-reduce-input | passes_local | 17.99149481791772 | 980 | 126 | 0.00180668522495 | promoted | Auto promoted after canonical re-score. |
| exp056 | impl_opt | opset13-reduce-attrs-fullpad | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp057 | impl_opt | opset12-attrs-fullpad | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp058 | impl_opt | reshape-known-color-vector | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp059 | impl_opt | reshape-known-color-vector-axis0 | passes_local | 17.99511801028714 | 978 | 124 | 0.00362319236942 | promoted | Auto promoted after canonical re-score. |
| exp060 | impl_opt | no-known-reshape-111 | passes_local | 17.97446168536148 | 1002 | 123 | 0.00266311741948 | promoted | Auto promoted after canonical re-score. |
| exp061 | impl_opt | rem0-u8-sum | passes_local | 17.98428757951277 | 993 | 121 | 0.00982589415129 | promoted | Auto promoted after canonical re-score. |
| exp062 | impl_opt | concat-remainder-rows | passes_local | 18.01250975299901 | 1011 | 72 | 0.0282221734862 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- Official ERROR repair: the 2026-06-12 single-task official submission for the previous model returned `SubmissionStatus.ERROR` despite local `passes_local` and `passes_rules`. The common pattern with other ERROR tasks was `TopK(UINT8)`. Repaired by keeping `color_counts10` as `FLOAT` through `TopK`; expected local score after repair is about `17.971798567941995`.
- Reached `17.99511801028714` by switching from dense selected 7x7 color-channel slices to a local-generator `rule_redesign`: fixed L-shape samples identify residue 1 and 2 colors, `TopK` on `UINT8` color counts supplies the three present colors, and residue 0 is selected as the remaining present color. The final output uses `Gather(rem_color_u8, remainder_index)` followed by `Equal(channel_ids_u8, selected_grid_u8)` and a zero `Pad`.
- Key improvements after exp041: exp042 used top-row/right-column L-shape scoring; exp043 replaced score matrices with fixed point `Max`; exp045 used only residue 1/2 fixed points plus remaining-color recovery; exp046 cast color counts to `UINT8` before `TopK`; exp049 kept `rem0` as a length-1 vector; exp051 cast known residue `ArgMax` outputs to `UINT8` before squeezing; exp055 moved to `Pad-18`, reused the int64 3-axis slice tensor for final `Pad`, and supplied `ReduceMax(rem0_terms)` axes as an input to avoid the opset18 reduction-attribute failure; exp059 reshaped known residue colors directly to length-1 vectors, removing the scalar squeeze/unsqueeze intermediates.
- Current task-local blocker: score 20 requires `memory_bytes_approx + params < 149`, while current total is `978 + 124 = 1102`. The pre-pad 9-channel bool tile `output9` alone is `441` bytes, the 8 required fixed point slices for two residues are `288` bytes, and `selected_grid_u8` is another `49` bytes before color counting and small control tensors. A set-cover recheck over all local examples confirms residue 1 needs the four points `(0,1),(0,4),(1,6),(4,6)` and residue 2 needs `(0,2),(0,5),(2,6),(5,6)`; no smaller fixed-point set covers the local generator distribution. 1-residue plus count/rank leaves the remaining two residues ambiguous. `TopK` values cannot be dropped, `ReduceSum(UINT8)` is unsupported, and the original `ArgMax` no-squeeze broadcast path breaks shape inference; exp058 also showed the reshape-vector variant still needs a separate `axis0` initializer for `ReduceMax`. exp056-exp057 confirmed that lowering opset to recover reduction/Squeeze axes attributes is not viable: opset13 rejects reduction axes attributes, while opset12 rejects `Pad(BOOL)`. Reaching 20 would require avoiding the dense one-hot pre-pad output or a substantially different scorer-compatible sparse output strategy.
- Revisited after task002 `Min`/direct-intermediate reductions: only the tiny remaining-color selection path would benefit, while the cost is dominated by `output9`, `selected_grid_u8`, and the fixed-point slices. No material path toward 20 was found without a different output construction.
- Revisited after task010 `UINT8 Add`, task002 `Sub`, and task001 `OneHot` checks: the remaining arithmetic path is only a 3-element color-set recovery, while `output9` and the fixed point slices dominate. `OneHot` is not implemented locally, and replacing `Gather(remainder_index)` plus `Equal(channel_ids_u8, selected_grid_u8)` with direct per-remainder masks would add larger 7x7 mask intermediates.
- Revisited after task010/task002 `ReduceSum` height/count wins: task007 already uses `ReduceSum(input, count_axes)` for color counts. The residue colors are recovered from fixed point slices; there is no valid-area or height-rank `ArgMax` to replace.
- Revisited after task006 compact-output audit: a 3-channel compact output cannot be used here because the three colors are dynamic example-specific channel ids. The fixed scorer channel order forces either the current 9-channel `Equal` pre-pad or a sparse/scatter-style channel placement that is not available in the supported ONNX subset.
