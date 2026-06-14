# task018 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 12.337959264188555 | 313588 | 1952 | 2026-06-14T23:47:35+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |
| dynamic-pad-slice-shift | impl_opt | expected_delta >=1.0: replace per-placement dense int32 source-index grids and Gather/Where shift with one zero-padded transformed color tensor per component/transform plus dynamic Slice per target; build failed because dynamic Slice output shape stayed symbolic. | build_failed |
| dynamic-pad-slice-shift-static-reshape | impl_opt | expected_delta >=1.0: keep the dynamic Pad/Slice shift but add a static Reshape to `[1,1,30,30]` after each Slice so public rules see fixed shapes; build still failed because raw Slice intermediates remained symbolic in inferred value_info. | build_failed |
| dynamic-pad-slice-shift-value-info | impl_opt | expected_delta >=1.0: keep dynamic Pad/Slice/Reshape and explicitly annotate raw Slice outputs as static `[1,1,30,30]` value_info to satisfy public-rule shape validation; this made the dynamic Slice shift public-compliant and cut dense shift memory. | promoted |
| direct-float-reducesum-casts | impl_opt | expected_delta 0.1-1.0: cast marker masks directly to FLOAT before ReduceSum instead of FLOAT16, avoiding ORT inserted precision-free FLOAT casts visible in the trace; build failed because prefix and per-placement count dtypes were inconsistent. | build_failed |
| direct-float-all-reducesum-casts | impl_opt | expected_delta 0.1-1.0: cast both component marker masks and per-placement marker match masks directly to FLOAT so ReduceSum and Equal count tensors share dtype; not better because explicit FLOAT masks cost more than the inserted precision-free cast path. | not_better |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp025 | impl_opt | f16-shared-shift-bool-output | passes_local | 11.669484889100698 | 614988 | 712 | 0.538750409061 | promoted | Auto promoted after canonical re-score. |
| zero001 | rule_redesign | zero-repair-size30 | passes_local | 11.004833100441234 | 1195752 | 1054 | -0.664651788659 | official_repaired | Official zero resolved: full 30x30 crop and 30-step growth scored 11.00 public. |
| exp026 | impl_opt | u8-maxpool-flat-shift | passes_local | 11.32244890984462 | 869180 | 1954 | 0.317615809403 | promoted | Auto promoted after canonical re-score. |
| exp027 | impl_opt | bool-base-u8-candidate-max | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp028 | impl_opt | bool-base-u8-candidate-max-v2 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp029 | impl_opt | bool-base-u8-candidate-max-v3 | passes_local | 11.498195212579247 | 728780 | 1954 | 0.175746302735 | promoted | Auto promoted after canonical re-score. |
| exp030 | impl_opt | color-counts-before-slice | passes_local | 11.54865810819162 | 692820 | 1954 | 0.0504628956124 | promoted | Auto promoted after canonical re-score. |
| exp031 | impl_opt | grow-steps-15-full30 | passes_local | 11.62956790036902 | 638820 | 1954 | 0.0809097921774 | promoted | Auto promoted after canonical re-score. |
| exp032 | impl_opt | grow-steps-8-full30 | passes_local | 11.669689555162744 | 613620 | 1954 | 0.0401216547937 | promoted | Auto promoted after canonical re-score. |
| exp033 | impl_opt | grow-steps-4-full30 | fails_local | 0.0 | 599220 | 1954 | -11.6696895552 | fails_local | Candidate did not pass local validation. |
| exp034 | impl_opt | conv-color-map | passes_local | 11.675558191186163 | 610008 | 1964 | 0.00586863602342 | promoted | Auto promoted after canonical re-score. |
| exp035 | impl_opt | drop-unused-cross-kernel | passes_local | 11.675572897849506 | 610008 | 1955 | 1.47066633431e-05 | promoted | Auto promoted after canonical re-score. |
| exp038 | impl_opt | hu-smoke-transform2 | fails_local | 0.0 | 368608 | 1955 | -11.6755728978 | fails_local | Candidate did not pass local validation. |
| exp039 | impl_opt | hu-smoke-grow4 | fails_local | 0.0 | 595608 | 1955 | -11.6755728978 | fails_local | Candidate did not pass local validation. |
| exp040 | impl_opt | hu-transform3-no-trans-vflip | fails_local | 0.0 | 489308 | 1955 | -11.6755728978 | fails_local | Candidate did not pass local validation. |
| exp041 | impl_opt | threshold01-drop-base-zero-check | passes_local | 11.787622259086902 | 545140 | 1955 | 0.112049361237 | promoted | Auto promoted after canonical re-score. |
| exp042 | impl_opt | threshold01-drop-source-overlap-check | passes_local | 11.88078156273975 | 496476 | 1954 | 0.0931593036528 | promoted | Auto promoted after canonical re-score. |
| exp043 | impl_opt | threshold01-marker-only-placement | passes_local | 12.007968567487396 | 436948 | 1954 | 0.127187004748 | promoted | Auto promoted after canonical re-score. |
| exp044 | impl_opt | threshold01-remove-noops-unused-counts | passes_local | 12.10935684866074 | 394644 | 1940 | 0.101388281173 | promoted | Auto promoted after canonical re-score. |
| exp045 | impl_opt | threshold01-color-match-only | fails_local | 0.0 | 351444 | 1940 | -12.1093568487 | fails_local | Candidate did not pass local validation. |
| exp046 | impl_opt | dynamic-pad-slice-shift | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp047 | impl_opt | dynamic-pad-slice-shift-static-reshape | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp048 | impl_opt | dynamic-pad-slice-shift-value-info | passes_local | 12.337959264188555 | 313588 | 1952 | 0.228602415528 | promoted | Auto promoted after canonical re-score. |
| exp049 | impl_opt | direct-float-reducesum-casts | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp050 | impl_opt | direct-float-all-reducesum-casts | passes_local | 12.24011120913886 | 346024 | 1952 | -0.0978480550497 | not_better | Passed but did not improve local_points. |

## Archived Summary
- None yet.
