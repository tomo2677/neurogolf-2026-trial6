# task028 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.921658420442327 | 1168 | 18 | 2026-06-13T10:49:54+09:00 | exp023 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | rule_redesign | fixed-row2-row7-colors | passes_local | 17.320286360033627 | 1926 | 238 | 1.42337554126 | promoted | Auto promoted after canonical re-score. |
| exp002 | rule_redesign | cols2-7-row-scan | passes_local | 17.46310287043383 | 1638 | 238 | 0.1428165104 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | row-fragments-template | passes_local | 17.60366470619919 | 1578 | 52 | 0.140561835765 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | pad-axes-after-fragments | passes_local | 17.604892453437515 | 1578 | 50 | 0.00122774723832 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | row-maxpool-no-axes | passes_local | 17.605506892780962 | 1578 | 49 | 0.000614439343448 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | u8-color-add | passes_local | 17.61414892187479 | 1564 | 49 | 0.00864202909383 | promoted | Auto promoted after canonical re-score. |
| exp008 | impl_opt | idx-background9 | passes_local | 17.616010542021492 | 1562 | 48 | 0.0018616201467 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | pooled-row-windows | passes_local | 17.747237581946813 | 1372 | 40 | 0.131227039925 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | top-both-color-diff-v2 | passes_local | 17.824510286375777 | 1265 | 42 | 0.077272704429 | promoted | Auto promoted after canonical re-score. |
| exp017 | impl_opt | actual-color-diff | passes_local | 17.850083163867893 | 1223 | 51 | 0.0255728774921 | promoted | Auto promoted after canonical re-score. |
| exp018 | impl_opt | sub-diff-edges | passes_local | 17.89093786431283 | 1191 | 32 | 0.0408547004449 | promoted | Auto promoted after canonical re-score. |
| exp019 | impl_opt | remove-zero-u8 | passes_local | 17.89175586026846 | 1191 | 31 | 0.000817995955632 | promoted | Auto promoted after canonical re-score. |
| exp020 | impl_opt | concat-full-rows | passes_local | 17.895034551730156 | 1191 | 27 | 0.00327869146169 | promoted | Auto promoted after canonical re-score. |
| exp021 | rule_redesign | no-same-color-fallback | passes_local | 17.90742628402532 | 1178 | 25 | 0.0123917322952 | promoted | Auto promoted after canonical re-score. |
| exp022 | impl_opt | u8-bottom-diff | passes_local | 17.915773577902083 | 1168 | 25 | 0.00834729387676 | promoted | Auto promoted after canonical re-score. |
| exp023 | impl_opt | zero-scalar-edge | passes_local | 17.921658420442327 | 1168 | 18 | 0.00588484254024 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
