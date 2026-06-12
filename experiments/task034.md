# task034 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.519470792955353 | 4074 | 746 | 2026-06-13T07:26:38+09:00 | exp008 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | direct-onehot-output9 | passes_local | 16.23344985045365 | 5176 | 1240 | 0.0220432817474 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | f16-conv-bool-nonzero | passes_local | 16.412907681204096 | 4123 | 1239 | 0.17945783075 | promoted | Auto promoted after canonical re-score. |
| exp003 | rule_redesign | public-range-kernel13 | passes_local | 16.506689749047094 | 4123 | 759 | 0.093782067843 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | maxpool-pad-cast | passes_local | 16.507304440184164 | 4123 | 756 | 0.00061469113707 | promoted | Auto promoted after canonical re-score. |
| exp006 | rule_redesign | kernel11-public-range | fails_local | 0.0 | 4123 | 564 | -16.5073044402 | fails_local | Candidate did not pass local validation. |
| exp007 | impl_opt | sparse-ray-weight | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp008 | impl_opt | subtract-marker-color | passes_local | 16.519470792955353 | 4074 | 746 | 0.0121663527712 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
