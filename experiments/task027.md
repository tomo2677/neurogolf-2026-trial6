# task027 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.505013766049466 | 1755 | 44 | 2026-06-13T10:52:17+09:00 | exp026 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | bool-rotation | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp002 | impl_opt | bool-rotation-logic-select | passes_local | 16.75199429839938 | 3743 | 77 | 0.336285233348 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | direct-onehot-output3 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp004 | impl_opt | direct-onehot-output3-opset18 | passes_local | 16.92597378387594 | 3143 | 67 | 0.173979485477 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | f16-overlap-counts | passes_local | 17.060484739337593 | 2739 | 67 | 0.134510955462 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | xor-black-channel | passes_local | 17.096773191269268 | 2639 | 67 | 0.0362884519317 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | where-select-rotation | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp009 | impl_opt | cast-blue-pad-axes | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp010 | impl_opt | cast-blue-only | passes_local | 17.09714280871942 | 2639 | 66 | 0.000369617450151 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | xor-select-rotation | passes_local | 17.097512562837146 | 2638 | 66 | 0.000369754117727 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | slice-reverse | passes_local | 17.10196030923538 | 2638 | 54 | 0.00444774639823 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | multi-axis-slice-reverse | passes_local | 17.16957438217967 | 2457 | 59 | 0.0676140729443 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | pad-rot10-inner | passes_local | 17.210545433913328 | 2367 | 48 | 0.0409710517337 | promoted | Auto promoted after canonical re-score. |
| exp016 | impl_opt | u8-overlap-counts | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp017 | impl_opt | inner-overlap10-count | passes_local | 17.23443091890268 | 2310 | 48 | 0.0238854849894 | promoted | Auto promoted after canonical re-score. |
| exp019 | rule_redesign | bbox-center-choice | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp020 | rule_redesign | bbox-center-choice-axes | passes_local | 17.434206717571485 | 1881 | 50 | 0.199775798669 | promoted | Auto promoted after canonical re-score. |
| exp022 | rule_redesign | col7-exception-probe | passes_local | 17.453025882483473 | 1838 | 57 | 0.018819164912 | promoted | Auto promoted after canonical re-score. |
| exp023 | rule_redesign | short-exception-gate | passes_local | 17.45461025038818 | 1836 | 56 | 0.00158436790471 | promoted | Auto promoted after canonical re-score. |
| exp024 | impl_opt | direct-rot10-slice | passes_local | 17.50279277679668 | 1755 | 48 | 0.0481825264085 | promoted | Auto promoted after canonical re-score. |
| exp025 | impl_opt | bool-reducemax | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp026 | impl_opt | pad-rot10-axes | passes_local | 17.505013766049466 | 1755 | 44 | 0.00222098925278 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
