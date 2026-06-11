# task006 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 19.989364705903746 | 126 | 24 | 2026-06-12T07:44:26+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | output3-final-pad | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp002 | impl_opt | output3-final-pad-opset11 | passes_local | 18.682835313252717 | 504 | 50 | 3.52805795766 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | output3-bool-final-pad | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp004 | impl_opt | output3-bool-final-pad-opset13 | passes_local | 19.52353644806849 | 198 | 41 | 0.840701134816 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | color-grid-output | passes_local | 19.9001335721758 | 135 | 29 | 0.376597124107 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | color-grid-axes3-slice | passes_local | 19.906249799193237 | 135 | 28 | 0.00611622701744 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | sparse-initializers | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp008 | impl_opt | direct-bool-templates | score_failed |  |  |  |  | score_failed | Candidate did not pass local validation. |
| exp009 | impl_opt | direct-u8-templates | passes_local | 19.956574883080755 | 126 | 29 | 0.0503250838875 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | sparse-u8-templates | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp011 | impl_opt | sparse-pad-direct-u8 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp012 | impl_opt | pad18-axes3 | passes_local | 19.969562078607566 | 126 | 27 | 0.0129871955268 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | min-u8-permuted-equal | passes_local | 19.989364705903746 | 126 | 24 | 0.0198026272962 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | sparse-colors3-equal | build_failed |  |  |  |  | build_failed | Candidate did not build. |

## Archived Summary
- Reached `19.989364705903746` by replacing the compact 10-channel output with a `BOOL` direct one-hot path: cast the two 3x3 blue slices to `UINT8`, use `Min(left_u8, right_u8)` for the overlap mask, then emit the 3-channel top-left grid with `Equal(colors3=[0,2,1], color2_top_u8)`. This removes the `black3/red3` templates while keeping `Pad-18` with shared `axes3`.
- Task-local blocker: 20 points requires `memory_bytes_approx + params < 149`, but the current total is `126 + 24 = 150`. The remaining static representation is already two necessary float input slices for left/right (`72` bytes), two compact `UINT8` casts plus `Min` (`27` bytes), and one compact 3-channel top-left `BOOL` grid (`27` bytes). The remaining 2 cost would have to come from `Slice`/`Pad`/`colors3` params or from replacing the compact 3-channel grid. The shared `axes3` already serves both `Slice` and `Pad`; recounting the initializer budget gives `left/right starts+ends=12`, `axes3=3`, `pads_output=6`, and `colors3=3`. Moving off opset18 increases full `Pad` pads, and omitting `Slice` axes increases starts/ends, so the current 24 params is the cheapest known wiring. Generating black/red channels with `Not/Concat`, replacing the compact grid with scalar color-grid logic, or using `OneHot`-style channel expansion saves params only by adding equal or larger memory. Sparse starts/pads/templates are checker-unsupported (`Slice`/`Pad`/`Where` reject sparse inputs), exp014 confirms `Equal` also rejects sparse `colors3`, `Where(BOOL)` is not implemented in ORT, and old-opset attribute `Slice`/`Pad` cannot keep this compact bool/uint8 output path. Re-audit after task003's permuted `Equal` trick found no remaining 2-cost reduction without a new final-output construction.
- Revisited after task002 `Min`/direct-seed reductions: the same idea is already present as `Min(left_u8,right_u8)`. Removing either input cast, the overlap tensor, or the 3-channel `Equal` output increases memory or breaks scorer semantics, so no 1-2 cost path was found.
- Revisited after task010 `UINT8 Add`, task002 `Sub`, and task001 `OneHot` checks: there is no scalar rank/seed subgraph here, `OneHot` is not implemented in the local ORT path, and compact output shapes cannot bypass the full `GRID_SHAPE` scorer comparison. `Mul(left_u8,right_u8)` is equivalent in cost to the current `Min`, while direct black/red channel construction needs extra `Cast/Not/Concat` tensors and is worse than `Equal(colors3, color2_top_u8)`.
- Revisited in the task002-010 cycle: wider single-slice wiring would save at most a few `Slice` params but adds larger float/uint8 intermediates; deriving right slice indices from left indices similarly trades 3-6 params for int tensor outputs that cost more. The remaining 150 total is therefore pinned by two `FLOAT` slices, two compact casts, `Min`, the 3-channel pre-pad `Equal`, and the minimal `Pad-18` axes wiring.
