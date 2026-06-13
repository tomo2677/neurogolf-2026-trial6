# task014 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 15.525527696032297 | 12949 | 74 | 2026-06-13T17:07:18+09:00 | exp014 |

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
| exp007 | impl_opt | window25-color-grid | passes_local | 14.879306488816416 | 24772 | 80 | 0.180159771204 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | argmax-u8-input | passes_local | 14.570246639210003 | 33772 | 80 | -0.309059849606 | not_better | Passed but did not improve local_points. |
| exp010 | impl_opt | combine-dynamic-output-pad | passes_local | 14.903745632808379 | 24179 | 73 | 0.024439143992 | promoted | Auto promoted after canonical re-score. |
| exp011 | impl_opt | pad-axes-dynamic-target | passes_local | 14.905024695553674 | 24147 | 74 | 0.0012790627453 | promoted | Auto promoted after canonical re-score. |
| exp012 | rule_redesign | min-count-target-bbox | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp013 | rule_redesign | min-count-target-bbox | passes_local | 15.525450911757678 | 12949 | 75 | 0.620426216204 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | drop-unused-one-i32 | passes_local | 15.525527696032297 | 12949 | 74 | 0.000076784274619 | promoted | Removed unused initializer after exp013 and re-scored canonical. |

## Archived Summary
- None yet.
