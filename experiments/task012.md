# task012 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.045978912721964 | 2782 | 65 | 2026-06-13T13:38:18+09:00 | exp019 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | nonzero-from-color12 | passes_local | 15.547576566890212 | 12564 | 175 | 0.341751359486 | promoted | Auto promoted after canonical re-score. |
| exp002 | rule_redesign | top2-dynamic-color-slices | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp003 | rule_redesign | top2-dynamic-color-slices-equal-split | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp004 | rule_redesign | top2-dynamic-color-slices-i32-axes | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp005 | rule_redesign | top2-dynamic-color-slices-valueinfo | passes_local | 16.045068139925302 | 7562 | 184 | 0.497491573035 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | h006 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp007 | impl_opt | h006 | passes_local | 16.21752373107546 | 6412 | 107 | 0.17245559115 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | bool-neighbor-center | passes_local | 16.2591833724327 | 6124 | 129 | 0.0416596413572 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | reuse-shift-up-pad | passes_local | 16.260463577440323 | 6124 | 121 | 0.00128020500762 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | opset18-pad-axes-v2 | passes_local | 16.26318946704611 | 6124 | 104 | 0.00272588960579 | promoted | Auto promoted after canonical re-score. |
| exp016 | rule_redesign | count-ranked-center | passes_local | 16.89711086535913 | 3214 | 90 | 0.633921398313 | promoted | Auto promoted after canonical re-score. |
| exp017 | impl_opt | direct-f16-center | passes_local | 16.941672693419044 | 3070 | 90 | 0.0445618280599 | promoted | Auto promoted after canonical re-score. |
| exp018 | impl_opt | weighted-single-conv | passes_local | 17.045627727468133 | 2782 | 66 | 0.103955034049 | promoted | Auto promoted after canonical re-score. |
| exp019 | impl_opt | cast-arm-fill | passes_local | 17.045978912721964 | 2782 | 65 | 0.00035118525383 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
