# task018 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 12.10935684866074 | 394644 | 1940 | 2026-06-14T23:40:35+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |
| zero-repair-size30 | rule_redesign | Official zero likely came from the `SIZE=24` internal crop and 8-step component growth shortcut; restore full 30x30 crop and 30-step growth before resubmit. | official_complete |
| threshold01-drop-source-overlap-check | impl_opt | expected_delta 0.1-1.0: remove the per-placement marker-source-overlap count and stale input0 path after base-zero removal, relying on target-anchor outside-source plus marker color/count checks; failure would show non-anchor markers can falsely land on the source component. | promoted |
| threshold01-marker-only-placement | impl_opt | expected_delta 0.1-1.0: remove per-placement shifted-count and inside-grid validation, relying on target marker count/color match to imply the transformed component is fully in-bounds; failure would show marker matches can accept partial or out-of-grid template placements. | promoted |
| threshold01-remove-noops-unused-counts | impl_opt | expected_delta 0.1-1.0: after the marker-only placement promotion, remove full-30 crop/no-op Pad plus unused component base/count tensors; failure would show one of these apparently dead dense tensors still affects public-rule output shape or invalid-cell masking. | promoted |
| threshold01-color-match-only | impl_opt | expected_delta 0.1-1.0: replace shifted-base exclusion plus marker-mask construction with shifted-nonzero exact color matches counted against marker_count; failure showed overlapping source/base-color cells can create false placements without explicit base filtering. | fails_local |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp015 | impl_opt | gather-1d-base-shift | passes_local | 9.138386119353912 | 7734904 | 2787 | 0.029341618218 | promoted | Auto promoted after canonical re-score. |
| exp016 | impl_opt | broadcast-shift-grids | passes_local | 9.190307494431744 | 7345144 | 1047 | 0.0519213750778 | promoted | Auto promoted after canonical re-score. |
| exp017 | impl_opt | int32-gather-indices | passes_local | 9.306584234263804 | 6538744 | 1047 | 0.116276739832 | promoted | Auto promoted after canonical re-score. |
| exp018 | rule_redesign | size24-internal-crop | passes_local | 9.74129999340989 | 4233460 | 719 | 0.434715759146 | promoted | Auto promoted after canonical re-score. |
| exp024 | impl_opt | color-grid-candidates | passes_local | 11.130734480039619 | 1054516 | 710 | 1.38943448663 | promoted | Auto promoted after canonical re-score. |
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

## Archived Summary
- None yet.
