# task021 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.364490581765722 | 5524 | 104 | 2026-06-13T04:23:57+09:00 | exp002 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | argmax-bg-bool-output | passes_local | 15.708263989819823 | 10744 | 104 | 1.44034702709 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | gather-bg-channel-bool-output | passes_local | 16.364490581765722 | 5524 | 104 | 0.656226591946 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
