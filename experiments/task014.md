# task014 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 14.699146717612358 | 29682 | 76 | 2026-06-13T05:30:35+09:00 | exp005 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | gathernd-u8-color-grid | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp002 | impl_opt | gathernd-u8-color-grid-shapeinfo | passes_local | 13.79047956794679 | 73734 | 96 | 1.12812418769 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | argmax-nonblack-u8 | passes_local | 14.44052211130028 | 38454 | 87 | 0.650042543353 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | reducemax-block-color | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp005 | impl_opt | reducemax-block-color | passes_local | 14.699146717612358 | 29682 | 76 | 0.258624606312 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
