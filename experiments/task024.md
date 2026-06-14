# task024 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 16.77986604284814 | 2790 | 925 | 2026-06-13T18:42:33+09:00 | exp018 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |
| hu-single-channel-line-conv | impl_opt | expected_delta 0.1-1.0: slice colors 1/3/2 first and use 1-channel line Conv weights, trading three 30x30 slices for fewer dense Conv params; not_better showed current dense Conv is still the better memory/param frontier. | not_better |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | rowcol-presence-first | passes_local | 15.748517684692642 | 10380 | 40 | 0.591193097832 | promoted | Auto promoted after canonical re-score. |
| exp002 | rule_redesign | public-max15-cap | passes_local | 15.512330967651527 | 13140 | 56 | -0.236186717041 | not_better | Passed but did not improve local_points. |
| exp003 | rule_redesign | public15-rowcol-presence | passes_local | 16.50423475599738 | 4830 | 64 | 0.755717071305 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | cast-presence | passes_local | 16.504439108710876 | 4830 | 63 | 0.000204352713496 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | reuse-15-starts | passes_local | 16.505256937421354 | 4830 | 59 | 0.000817828710478 | promoted | Auto promoted after canonical re-score. |
| exp011 | impl_opt | direct-color2-columns | passes_local | 16.568582585605167 | 4530 | 59 | 0.0633256481838 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | colors4-bool-pad | passes_local | 16.569890915490873 | 4530 | 53 | 0.00130832988571 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | slice-axes-hw-chw | passes_local | 16.571419466940366 | 4530 | 46 | 0.00152855144949 | promoted | Auto promoted after canonical re-score. |
| exp017 | impl_opt | conv-sparse-line-presence | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp018 | impl_opt | conv-dense-line-presence | passes_local | 16.77986604284814 | 2790 | 925 | 0.208446575908 | promoted | Auto promoted after canonical re-score. |
| exp019 | impl_opt | hu-single-channel-line-conv | passes_local | 15.472734337102395 | 13590 | 139 | -1.30713170575 | not_better | Passed but did not improve local_points. |
| exp020 | impl_opt | sparse-conv-debug | build_failed |  |  |  |  | build_failed | Candidate did not build. |

## Archived Summary
- None yet.
