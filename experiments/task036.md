# task036 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 13.77180097965592 | 75045 | 177 | 2026-06-13T07:36:41+09:00 | exp007 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp002 | impl_opt | h001 | passes_local | 12.928138978399213 | 172953 | 1928 | 0.748085592345 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | h002 | passes_local | 12.92819044329016 | 172953 | 1919 | 5.1464890948e-05 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | f16-density-count | passes_local | 13.025518990574064 | 156735 | 1919 | 0.0973285472839 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | int32-crop-indices | passes_local | 13.145827999248258 | 138751 | 1919 | 0.120309008674 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | crop5-window | passes_local | 13.47388194124379 | 101151 | 177 | 0.328053941996 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | f16-full-input-density | passes_local | 13.77180097965592 | 75045 | 177 | 0.297919038412 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
