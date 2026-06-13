# task018 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 9.306584234263804 | 6538744 | 1047 | 2026-06-13T09:24:45+09:00 | exp017 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | passes_local | 8.298108748965113 | 17925504 | 2800 |  | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | h001 | passes_local | 8.298108748965113 | 17925504 | 2800 | 0 | not_better | Passed but did not improve local_points. |
| exp003 | impl_opt | h002 | passes_local | 8.946564904291776 | 9371056 | 2800 | 0.648456155327 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | grow-steps-15 | passes_local | 8.975791382000043 | 9101056 | 2800 | 0.0292264777083 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | grow-steps-8 | passes_local | 8.989728341011567 | 8975056 | 2800 | 0.0139369590115 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | int32-static-shift | passes_local | 9.056021454579346 | 8399184 | 2801 | 0.0662931135678 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | where-shift-bounds | passes_local | 9.069827364755408 | 8283984 | 2801 | 0.0138059101761 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | dedupe-initializers | passes_local | 9.069829054193692 | 8283984 | 2787 | 1.68943828349e-06 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | remove-transform-identities | passes_local | 9.109044501135914 | 7965304 | 2787 | 0.0392154469422 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | gather-1d-base-shift | passes_local | 9.138386119353912 | 7734904 | 2787 | 0.029341618218 | promoted | Auto promoted after canonical re-score. |
| exp016 | impl_opt | broadcast-shift-grids | passes_local | 9.190307494431744 | 7345144 | 1047 | 0.0519213750778 | promoted | Auto promoted after canonical re-score. |
| exp017 | impl_opt | int32-gather-indices | passes_local | 9.306584234263804 | 6538744 | 1047 | 0.116276739832 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
