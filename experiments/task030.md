# task030 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 15.177180350531327 | 18208 | 242 | 2026-06-13T07:53:52+09:00 | exp005 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | u8-colorgrid-output | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp002 | impl_opt | u8-colorgrid-output-ssa | passes_local | 13.060448245753605 | 151376 | 1832 | 0.290598680763 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | int32-shift-indices | passes_local | 13.297934609566152 | 118988 | 1833 | 0.237486363813 | promoted | Auto promoted after canonical re-score. |
| exp004 | rule_redesign | window10-shift | passes_local | 15.1769635718615 | 18208 | 246 | 1.87902068555 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | reuse-shape-width | passes_local | 15.177180350531327 | 18208 | 242 | 0.000216778669827 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
