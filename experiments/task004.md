# task004 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 15.702381619916757 | 10841 | 71 | 2026-06-12T07:22:58+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp020 | impl_opt | top3-color-ids-unsqueeze | passes_local | 14.969527435709383 | 22618 | 90 | 4.40363740513e-05 | promoted | Auto promoted after canonical re-score. |
| exp021 | impl_opt | pad-color-grid-output | passes_local | 15.04543917727987 | 20958 | 90 | 0.0759117415705 | promoted | Auto promoted after canonical re-score. |
| exp022 | impl_opt | opset12-unsqueeze-attrs | passes_local | 15.04558171879533 | 20958 | 87 | 0.000142541515459 | promoted | Auto promoted after canonical re-score. |
| exp023 | impl_opt | topk-flat-present | passes_local | 15.046770355569057 | 20934 | 86 | 0.00118863677373 | promoted | Auto promoted after canonical re-score. |
| exp024 | impl_opt | drop-nonblack-any | passes_local | 15.0839911669102 | 20166 | 86 | 0.0372208113411 | promoted | Auto promoted after canonical re-score. |
| exp025 | impl_opt | dynamic-selected-slices | passes_local | 15.44132940462065 | 14066 | 101 | 0.35733823771 | promoted | Auto promoted after canonical re-score. |
| exp026 | impl_opt | int32-dynamic-slice-indices | passes_local | 15.44983587116909 | 13946 | 101 | 0.00850646654844 | promoted | Auto promoted after canonical re-score. |
| exp027 | impl_opt | topk-empty-values | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp028 | impl_opt | u8-present-topk | passes_local | 15.45168851527887 | 13920 | 101 | 0.00185264410978 | promoted | Auto promoted after canonical re-score. |
| exp029 | impl_opt | derive-channel-end | passes_local | 15.452330615682623 | 13920 | 92 | 0.000642100403754 | promoted | Auto promoted after canonical re-score. |
| exp030 | impl_opt | selected-slice-axes3-revisit | passes_local | 15.453973414523569 | 13896 | 93 | 0.00164279884095 | promoted | Auto promoted after canonical re-score. |
| exp031 | impl_opt | per-slot-pipeline | passes_local | 15.510438246489903 | 13128 | 93 | 0.0564648319663 | promoted | Auto promoted after canonical re-score. |
| exp032 | impl_opt | colorize-before-shift | passes_local | 15.510438246489903 | 13128 | 93 | 0 | not_better | Passed but did not improve local_points. |
| exp033 | impl_opt | float-rowcol-colorized | passes_local | 15.54741958101862 | 12648 | 93 | 0.0369813345287 | promoted | Auto promoted after canonical re-score. |
| exp034 | impl_opt | rect-valid-from-input | passes_local | 15.631716149957862 | 11624 | 87 | 0.0842965689392 | promoted | Auto promoted after canonical re-score. |
| exp035 | impl_opt | keep-over-shift | fails_local | 0.0 | 11880 | 93 | -15.63171615 | fails_local | Candidate did not pass local validation. |
| exp036 | impl_opt | rect-valid-float-rowcol | passes_local | 15.673566909036964 | 11144 | 87 | 0.0418507590791 | promoted | Auto promoted after canonical re-score. |
| exp037 | impl_opt | rect-valid-colorized | passes_local | 15.673566909036964 | 11144 | 87 | 0 | not_better | Passed but did not improve local_points. |
| exp038 | impl_opt | slice-end-delta | passes_local | 15.674546820923313 | 11132 | 88 | 0.000979911886349 | promoted | Auto promoted after canonical re-score. |
| exp039 | impl_opt | cast-channel-id-color | passes_local | 15.675349281846406 | 11132 | 79 | 0.000802460923094 | promoted | Auto promoted after canonical re-score. |
| exp040 | impl_opt | topidx-plus-one-channel-id | passes_local | 15.676063121441278 | 11132 | 71 | 0.000713839594871 | promoted | Auto promoted after canonical re-score. |
| exp041 | impl_opt | rowcol-from-selected-color | passes_local | 15.702106731015956 | 10844 | 71 | 0.0260436095747 | promoted | Auto promoted after canonical re-score. |
| exp042 | impl_opt | broadcast-slot-color-id | passes_local | 15.702381619916757 | 10841 | 71 | 0.000274888900801 | promoted | Auto promoted after canonical re-score. |
| exp043 | impl_opt | opset18-pad-axes | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp044 | impl_opt | valid-area-rowcol-count | build_failed |  |  |  |  | build_failed | Candidate did not build. |

## Archived Summary
- Official ERROR repair: the 2026-06-12 single-task official submission for the previous model returned `SubmissionStatus.ERROR` despite local `passes_local` and `passes_rules`. The common pattern with other ERROR tasks was `TopK(UINT8)`. Repaired by keeping the present-color score vector as `FLOAT` through `TopK`; expected local score after repair is about `15.700001756093283`.
- Reached `15.702381619916757` / `10841` bytes / `71` params with a per-slot `UINT8` color-grid pipeline: use `TopK(k=3)` over present nonblack colors, dynamically slice only those 3 channels in the 16x16 active window, colorize each selected channel before the shift, apply bottom/right caps per slot, merge slot color grids, pad a single `UINT8 [1,1,30,30]` color grid with invalid `255`, then make `Equal(colors10, color30)` the graph output.
- Recent improvements after exp030: exp031 switched from a vectorized `[1,3,16,16]` selected-channel pipeline to independent slots; exp033 used the float selected slice for row/col reductions and colorized before shifting; exp034 replaced dense black/selected valid-area reconstruction with a rectangular valid mask derived from full-input row/col presence; exp036 combined rectangular valid with the float-row/col colorized path; exp038-exp040 trimmed dynamic slice/color-ID params; exp041 reuses `selected_color` for row/col presence, saving another `288` bytes; exp042 broadcasts scalar slot color IDs directly and removes three `Unsqueeze` outputs.
- Current task-local blocker: 20 points requires `memory+params <= 148`, but the passing strategy has a dense lower bound orders of magnitude above that. The 3 dynamic selected-channel float slices alone are `3 * 1024` bytes, `color30` is `900` bytes, and each selected color still needs dense 16x16 `selected_bool`, `selected_color`, `keep_mask`, `kept`, `shift_source`, `shifted`, and `slot_color` tensors. A second-pass audit found no unused initializers or cheaper `TopK`/`Slice` output wiring. Local data has 47 examples with 3 nonblack colors, so `k=2` is invalid; shapes are not square and output nonzeros can hit row/col 15, so task002/task009-style square-valid or smaller-crop shortcuts do not apply. exp035 showed dropping the explicit kept/shift merge loses shifted cells landing on empty keep-mask positions; exp043 showed moving to opset18 for `Pad` axes is not useful because all `ReduceMax` axes attributes would also need input tensors, offsetting the shorter pads. Older alternatives found `OneHot`, BOOL `ReduceMax`, sparse/dynamic output construction, and `Mul/Sub(uint8)` unavailable or unsupported. Further large gains require avoiding dense per-color 16x16 materialization or introducing a sparse/dynamic per-object representation not available under the current scorer/operator constraints.
- Revisited after task002 `MaxPool(UINT8)` line-of-sight wins: not applicable to the dominant part of task004. Its bbox work is already compact `ReduceMax/ArgMax`; cost is dominated by dynamic per-color 16x16 materialization and slot composition.
- Revisited after task002 `Sub(one_u8, mask)` and task001 `OneHot` checks: task004 does not have a removable `Equal(...)->Cast(UINT8)` 0/1 seed in the dominant path; the repeated tensors are color-valued selected/kept/shifted grids, not just binary masks. `OneHot` is also unavailable in the local ORT path, and final compact-output tricks are blocked by the full `GRID_SHAPE` scorer comparison.
- Revisited after task010/task002 `ArgMax -> ReduceSum` wins: exp044 attempted row/col count valid-area construction, but `row_idx`/`col_idx` are also reused for per-object `Equal` against `ArgMax(INT64)` bottom/right indices. Converting them to `FLOAT` breaks type checking, while adding duplicate float indices or casting all object indices costs more than the two valid-area `ArgMax` tensors save.
- Revisited after the task001/task003 dense-output blockers: omitting the rectangular valid mask is still invalid because absent cells outside the variable input rectangle must be all-false in the 30x30 comparison, not black. A color-grid/`ArgMax` rewrite would need a dense multi-channel input crop before reducing colors, so it does not avoid the current `3 * 16 * 16` selected-channel lower bound.
