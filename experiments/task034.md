# task034 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.506689749047094 | 4123 | 759 | 2026-06-13T03:54:20+09:00 | exp003 |

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

## Archived Summary
- None yet.
