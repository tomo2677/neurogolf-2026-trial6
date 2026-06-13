# task021 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.836059045244994 | 3430 | 82 | 2026-06-13T13:08:06+09:00 | exp016 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | argmax-bg-bool-output | passes_local | 15.708263989819823 | 10744 | 104 | 1.44034702709 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | gather-bg-channel-bool-output | passes_local | 16.364490581765722 | 5524 | 104 | 0.656226591946 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | reduce-before-gather | passes_local | 16.604296706171475 | 4324 | 104 | 0.239806124406 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | reuse-prev-starts | passes_local | 16.60520045679783 | 4324 | 100 | 0.000903750626357 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | single-pad-prev-shift | passes_local | 16.621149758205508 | 4266 | 88 | 0.0159493014077 | promoted | Auto promoted after canonical re-score. |
| exp008 | impl_opt | bg-row-then-col | passes_local | 16.769422782853546 | 3666 | 88 | 0.148273024648 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | f16-dimension-sum | fails_local | 0.0 | 3542 | 88 | -16.7694227829 | fails_local | Candidate did not pass local validation. |
| exp011 | impl_opt | opset18-bool-prev-pad | passes_local | 16.799711739712446 | 3546 | 96 | 0.0302889568589 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | f16-start-sum-cast | passes_local | 16.832080637042182 | 3430 | 96 | 0.0323688973297 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | prev-pad-axes | passes_local | 16.835489731252956 | 3430 | 84 | 0.00340909421077 | promoted | Auto promoted after canonical re-score. |
| exp016 | impl_opt | reuse-prev-pads | passes_local | 16.836059045244994 | 3430 | 82 | 0.000569313992038 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
