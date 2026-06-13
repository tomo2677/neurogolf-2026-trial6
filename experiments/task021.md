# task021 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.769422782853546 | 3666 | 88 | 2026-06-13T10:30:02+09:00 | exp008 |

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

## Archived Summary
- None yet.
