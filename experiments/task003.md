# task003 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 19.531939858864867 | 218 | 19 | 2026-06-12T07:43:53+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp008 | impl_opt | rowpair-gather-slice4 | passes_local | 19.07574420258547 | 288 | 86 | 0.0771590805466 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | rowpair-gather-row-gather | passes_local | 19.153561224942276 | 288 | 58 | 0.0778170223568 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | color-grid-output | passes_local | 19.21925648420767 | 288 | 36 | 0.0656952592654 | promoted | Auto promoted after canonical re-score. |
| exp011 | impl_opt | split-rows | passes_local | 19.231679004206228 | 288 | 32 | 0.0124225199986 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | row-eq-u8-scalar | passes_local | 19.24109822612272 | 285 | 32 | 0.00941922191649 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | full-color-grid-output | passes_local | 17.957713828060257 | 1104 | 40 | -1.28338439806 | not_better | Passed but did not improve local_points. |
| exp014 | impl_opt | where-period | passes_local | 19.269900217026425 | 275 | 33 | 0.0288019909037 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | direct-bool-output3 | score_failed |  |  |  |  | score_failed | Candidate did not pass local validation. |
| exp016 | impl_opt | direct-u8-output3 | passes_local | 19.358092929061886 | 248 | 34 | 0.0881927120355 | promoted | Auto promoted after canonical re-score. |
| exp017 | rule_redesign | first6-plus-first3 | fails_local | 0.0 | 198 | 31 | -19.3580929291 | fails_local | Candidate did not pass local validation. |
| exp018 | impl_opt | direct-row-select | score_failed |  |  |  |  | score_failed | Candidate did not pass local validation. |
| exp019 | impl_opt | direct-row-select-u8 | passes_local | 19.41275134159975 | 245 | 22 | 0.0546584125379 | promoted | Auto promoted after canonical re-score. |
| exp020 | impl_opt | sparse-output-constants | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp021 | impl_opt | opset9-slice-pad-attrs | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp022 | impl_opt | opset10-pad-attr | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp023 | impl_opt | opset9-bool-rows | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp024 | impl_opt | opset9-bool-mux | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp025 | impl_opt | bool-rows-final-u8 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp026 | impl_opt | bool-rows-final-u8-axis | score_failed |  |  |  |  | score_failed | Candidate did not pass local validation. |
| exp027 | impl_opt | bool-reducemin | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp028 | impl_opt | bool-reducemin-period | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp029 | impl_opt | equal-output3-color | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp030 | impl_opt | permuted-equal-output3 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp031 | impl_opt | permuted-equal-output3-bool | passes_local | 19.531939858864867 | 218 | 19 | 0.119188517265 | promoted | Auto promoted after canonical re-score. |
| exp032 | impl_opt | row-code-matmul-period | build_failed |  |  |  |  | build_failed | Candidate did not build. |

## Archived Summary
- task003 local cost was improved from `14.912151305372147` / `23888` bytes / `161` params to `19.531939858864867` / `218` bytes / `19` params.
- The current best emits a compact `BOOL` one-hot pre-pad tensor with `Equal(colors3=[0,2,1], red_top_u8)`, then pads to the full NeuroGolf output. exp019 removed the dynamic `Mod(arange9, period)`/`Gather` path and directly selects only rows 4-8 from period booleans; exp031 reused the existing `red_top_u8` `0/1` grid as a color code and removed the `Cast(red_top_u8 -> BOOL)` plus `black3/red3` templates.
- Failed alternatives: exp017's "first 6 rows + first 3 rows" shortcut fails p4 examples; exp018 confirms `Where` with `BOOL` X/Y is unsupported by ORT; exp020 shows sparse initializers remain `sparse_tensor` inputs and cannot feed `Where`; exp021-exp024 show older opsets cannot combine these tricks because `Equal(UINT8)`, `ReduceMin(UINT8)`, or `Pad(UINT8)` is unsupported; exp027 shows `ReduceMin(BOOL)` is unsupported by ONNX checker; exp029-exp030 show the permuted-color trick must keep the graph output as `BOOL`; exp032 shows `MatMul(UINT8)` cannot be used to compress each row into a scalar period code.
- Current task-local blocker: 20 points requires `memory+params <= 148`, but this output strategy has a hard lower bound above that before period detection: `output3` (`BOOL [1,3,9,3]`, 81 bytes), `red_top_u8` (`UINT8 [1,1,9,3]`, 27 bytes), the required input slice (`FLOAT [1,1,4,3]`, 48 bytes), and its compact cast (`UINT8 [1,1,4,3]`, 12 bytes) already total `168` bytes. Returning a non-one-hot output or dropping the black channel fails scorer semantics after `run_network` thresholding, and the color-grid/`Equal(colors, grid)` alternative was measured much worse at `1104` bytes / `40` params. Revisited after task002 final-output experiments: a 2-channel pre-pad cannot work because `Pad` cannot insert the false channel between black channel 0 and red channel 2, and using a pad constant to synthesize black also fills unwanted spatial/channel regions. Data-specific single-column/two-column period shortcuts fail examples, and simple p2 shortcuts (`eq13`, `eq02`, `eq02 && !eq03`, `eq13 && !eq03`) still leave local mismatches, so the current 3-wide `eq02 && eq13` check cannot be collapsed. Further progress likely requires a different final-output construction that can expand a 1-channel red mask into black/channel2 one-hot while using final output exemption; current supported `Pad`, `Where`, and reduction type constraints do not provide that.
- Revisited after task002 line-of-sight `MaxPool(UINT8)` improvements: not applicable because task003 has no flood/visibility seed; its cost is dominated by compact periodic row selection plus the unavoidable 3-channel one-hot pre-pad output.
- Revisited after returning from task001/task002 blockers: the scorer's full 30x30 `np.array_equal` requirement means the black channel cannot be omitted, and `Pad` cannot synthesize a false channel between channel 0 and channel 2 from a 2-channel tensor. The current `Equal(colors3=[0,2,1], red_top_u8)` remains the cheapest known way to create black plus red while keeping channel 1 false.
- Revisited after task010 `UINT8 Add` and task001 `OneHot` checks: task003 has no scalar rank/flood seed where `Add/Sub` removes a 20x20 cast; `OneHot` is not implemented in the local ORT path; and compact `[1,3,9,3]` graph output cannot be used because local scoring compares against full `GRID_SHAPE` one-hot tensors. The hard lower bound from `output3 + red_top_u8 + input slice/cast` still exceeds the 20-point target.
- Revisited after task010/task002 `ArgMax -> ReduceSum` wins: task003 has no valid-area count or height-rank scalar to replace. Its remaining control path is already small row equality on four 1x3 rows; the dominant cost is still the compact 3-channel pre-pad output and input slice/cast.
