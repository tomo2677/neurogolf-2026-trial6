# task019 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.849243529724447 | 3389 | 77 | 2026-06-13T10:43:23+09:00 | exp019 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | passes_local | 11.249227449736084 | 935388 | 1925 |  | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | h002 | passes_local | 11.264734670215155 | 920988 | 1902 | 0.0155072204791 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | colorgrid-dynamic-shift | passes_local | 12.74435654270408 | 208264 | 1900 | 1.47962187249 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | f16-diag-conv | passes_local | 12.761634429273563 | 204664 | 1900 | 0.0172778865695 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | int32-shift-indices | passes_local | 13.06455567235578 | 150680 | 1900 | 0.302921243082 | promoted | Auto promoted after canonical re-score. |
| exp006 | rule_redesign | twelve-window-shift | passes_local | 14.84914468592548 | 25240 | 373 | 1.78458901357 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | reuse-shape-width | passes_local | 14.849300868815961 | 25240 | 369 | 0.00015618289048 | promoted | Auto promoted after canonical re-score. |
| exp008 | impl_opt | mod-tile-gather | passes_local | 15.839900844460564 | 9148 | 362 | 0.990599975645 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | gather-1d-tile | passes_local | 15.919654424640004 | 8428 | 353 | 0.0797535801794 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | broadcast-tile-grids | passes_local | 16.15595210105751 | 6844 | 89 | 0.236297676418 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | int32-tile-gather | passes_local | 16.337668042917525 | 5692 | 89 | 0.18171594186 | promoted | Auto promoted after canonical re-score. |
| exp015 | rule_redesign | single-color-zero-mask | passes_local | 16.596199495938855 | 4361 | 103 | 0.258531453021 | promoted | Auto promoted after canonical re-score. |
| exp016 | impl_opt | single-color-axis-gather | passes_local | 16.719542313417442 | 3869 | 77 | 0.123342817479 | promoted | Auto promoted after canonical re-score. |
| exp017 | impl_opt | diag-fill-direct | passes_local | 16.79532817104919 | 3581 | 77 | 0.0757858576317 | promoted | Auto promoted after canonical re-score. |
| exp018 | impl_opt | raw-colored | fails_local | 0.0 | 3437 | 77 | -16.795328171 | fails_local | Candidate did not pass local validation. |
| exp019 | rule_redesign | bg-shape | passes_local | 16.849243529724447 | 3389 | 77 | 0.0539153586753 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
