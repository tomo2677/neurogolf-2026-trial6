# task018 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 11.675572897849506 | 610008 | 1955 | 2026-06-13T17:35:04+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |
| zero-repair-size30 | rule_redesign | Official zero likely came from the `SIZE=24` internal crop and 8-step component growth shortcut; restore full 30x30 crop and 30-step growth before resubmit. | official_complete |
| conv-color-map | impl_opt | Replace full-grid ArgMax color decoding with a 1x1 FLOAT Conv and compare base/anchor colors as UINT8 to reduce INT64 memory. | promoted |
| drop-unused-cross-kernel | impl_opt | Remove the unused cross_kernel initializer left after the current component-growth implementation. | promoted |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp005 | impl_opt | grow-steps-8 | passes_local | 8.989728341011567 | 8975056 | 2800 | 0.0139369590115 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | int32-static-shift | passes_local | 9.056021454579346 | 8399184 | 2801 | 0.0662931135678 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | where-shift-bounds | passes_local | 9.069827364755408 | 8283984 | 2801 | 0.0138059101761 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | dedupe-initializers | passes_local | 9.069829054193692 | 8283984 | 2787 | 1.68943828349e-06 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | remove-transform-identities | passes_local | 9.109044501135914 | 7965304 | 2787 | 0.0392154469422 | promoted | Auto promoted after canonical re-score. |
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

## Archived Summary
- None yet.
