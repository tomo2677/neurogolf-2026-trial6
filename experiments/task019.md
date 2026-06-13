# task019 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.15595210105751 | 6844 | 89 | 2026-06-13T09:09:45+09:00 | exp013 |

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

## Archived Summary
- None yet.
