# task030 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.83833954794372 | 3448 | 56 | 2026-06-13T13:10:04+09:00 | exp020 |

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
| exp006 | impl_opt | gather-1d-shift | passes_local | 15.262330932158813 | 16708 | 236 | 0.0851505816275 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | direct-c1-mask | passes_local | 15.459349193516413 | 13678 | 236 | 0.197018261358 | promoted | Auto promoted after canonical re-score. |
| exp008 | impl_opt | direct-c1-color | passes_local | 15.47382751215569 | 13478 | 236 | 0.0144783186393 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | remove-unused-one-u8 | passes_local | 15.473900433000146 | 13478 | 235 | 7.29208444561e-05 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | broadcast-shift-grids | passes_local | 15.709647690054432 | 10778 | 55 | 0.235747257054 | promoted | Auto promoted after canonical re-score. |
| exp011 | impl_opt | int32-shift-gather | passes_local | 15.869460698227373 | 9178 | 55 | 0.159813008173 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | opset18-pad-axes | passes_local | 15.869677336009849 | 9178 | 53 | 0.000216637782476 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | colors5-valid-pad | passes_local | 15.914089706401993 | 8778 | 52 | 0.0444123703921 | promoted | Auto promoted after canonical re-score. |
| exp016 | rule_redesign | direct-0124-masks | passes_local | 16.453053850434415 | 5078 | 73 | 0.538964144032 | promoted | Auto promoted after canonical re-score. |
| exp017 | impl_opt | row-valid-col0 | passes_local | 16.583511512705392 | 4448 | 73 | 0.130457662271 | promoted | Auto promoted after canonical re-score. |
| exp018 | impl_opt | slice-chw-axes | passes_local | 16.584618074730447 | 4448 | 68 | 0.00110656202505 | promoted | Auto promoted after canonical re-score. |
| exp019 | impl_opt | gather-axis2-shift | passes_local | 16.83776893451882 | 3448 | 58 | 0.253150859788 | promoted | Auto promoted after canonical re-score. |
| exp020 | impl_opt | final-pad-axes | passes_local | 16.83833954794372 | 3448 | 56 | 0.000570613424898 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
