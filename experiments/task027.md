# task027 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.10196030923538 | 2638 | 54 | 2026-06-13T08:40:44+09:00 | exp013 |

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

## Archived Summary
- None yet.
