# task027 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.096773191269268 | 2639 | 67 | 2026-06-13T03:20:14+09:00 | ledger |

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

## Archived Summary
- None yet.
