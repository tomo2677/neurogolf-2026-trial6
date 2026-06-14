# task036 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 15.497437875530004 | 13343 | 51 | 2026-06-13T19:08:50+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |
| conv-color-map | impl_opt | Replace full-grid ArgMax color decoding with a 1x1 FLOAT Conv over one-hot channels to reduce INT64 memory. | promoted |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp007 | impl_opt | f16-full-input-density | passes_local | 13.77180097965592 | 75045 | 177 | 0.297919038412 | promoted | Auto promoted after canonical re-score. |
| exp008 | rule_redesign | right-pair-target | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp009 | rule_redesign | right-pair-target-opset13 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp010 | rule_redesign | right-pair-target-slice | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp011 | rule_redesign | right-pair-target-slice-axes | passes_local | 13.900108212637889 | 65771 | 393 | 0.128307232982 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | right-pair-pad13 | passes_local | 14.046197944203515 | 57071 | 100 | 0.146089731566 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | reuse-one-i64 | passes_local | 14.046232927586411 | 57071 | 98 | 3.49833828963e-05 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | dynamic-target-slice | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp016 | impl_opt | dynamic-target-slice-valueinfo | passes_local | 14.180501754797035 | 49887 | 99 | 0.134268827211 | promoted | Auto promoted after canonical re-score. |
| exp017 | impl_opt | reuse-shape1 | passes_local | 14.180521760598719 | 49887 | 98 | 2.00058016837e-05 | promoted | Auto promoted after canonical re-score. |
| exp018 | impl_opt | gather-1d-crop | passes_local | 14.183206164630677 | 49762 | 89 | 0.00268440403196 | promoted | Auto promoted after canonical re-score. |
| exp019 | impl_opt | broadcast-crop-grids | passes_local | 14.188032137681693 | 49562 | 49 | 0.00482597305102 | promoted | Auto promoted after canonical re-score. |
| exp020 | impl_opt | int32-crop-gather | passes_local | 14.192071649546953 | 49362 | 49 | 0.00403951186526 | promoted | Auto promoted after canonical re-score. |
| exp021 | impl_opt | bool-gather-crop | passes_local | 14.1925777377993 | 49337 | 49 | 0.000506088252347 | promoted | Auto promoted after canonical re-score. |
| exp022 | impl_opt | bool-crop-pad | passes_local | 14.20580622580255 | 48687 | 50 | 0.0132284880033 | promoted | Auto promoted after canonical re-score. |
| exp023 | impl_opt | default-false-pad | passes_local | 14.205826744305112 | 48687 | 49 | 2.0518502561e-05 | promoted | Auto promoted after canonical re-score. |
| exp025 | impl_opt | u8-pair-sum | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp026 | impl_opt | axis-gather-crop | passes_local | 14.228047345072321 | 47617 | 48 | 0.0222206007672 | promoted | Auto promoted after canonical re-score. |
| exp027 | impl_opt | pad-axes-opset18 | passes_local | 14.22817323152291 | 47617 | 42 | 0.000125886450588 | promoted | Auto promoted after canonical re-score. |
| exp029 | rule_redesign | priority-strong-pair-color | passes_local | 14.6748459600177 | 30445 | 45 | 0.446672728495 | promoted | Auto promoted after canonical re-score. |
| exp030 | impl_opt | priority-strong-pair-nomask | passes_local | 14.906305396395927 | 24145 | 45 | 0.231295434738 | promoted | Auto promoted after canonical re-score. |
| exp031 | impl_opt | drop-unused-initializers-final | passes_local | 14.906512114742917 | 24145 | 40 | 0.00020671834699 | promoted | Canonical re-score after removing unused initializers. |
| exp032 | impl_opt | conv-color-map | passes_local | 15.06719639312982 | 20545 | 50 | 0.160684278387 | promoted | Auto promoted after canonical re-score. |
| exp035 | impl_opt | lshape-hpair-min-target | passes_local | 15.497437875530004 | 13343 | 51 | 0.4302414824 | promoted | Auto promoted after canonical re-score. |
| exp036 | impl_opt | hu-smoke-out4 | fails_local | 0.0 | 13161 | 49 | -15.4974378755 | fails_local | Candidate did not pass local validation. |

## Archived Summary
- None yet.
