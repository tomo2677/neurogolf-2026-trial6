# task025 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 13.672524981698361 | 83012 | 61 | 2026-06-13T17:29:20+09:00 | exp017 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |
| conv-color-map | impl_opt | Replace full-grid ArgMax color decoding with a 1x1 FLOAT Conv over one-hot channels to reduce INT64 memory. | promoted |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | passes_local | 12.285490581737783 | 331868 | 670 | 0.933007249924 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | dedupe-initializers | passes_local | 12.287055536703585 | 331868 | 150 | 0.0015649549658 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | color-grid-masks-f16-counts | passes_local | 12.315304725743935 | 322692 | 78 | 0.0282491890404 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | single-pad-shifts | passes_local | 12.417458606803253 | 291372 | 54 | 0.102153881059 | promoted | Auto promoted after canonical re-score. |
| exp008 | impl_opt | bool-pad-or-opset13-sumaxes | passes_local | 12.50450180276167 | 267072 | 60 | 0.0870431959584 | promoted | Auto promoted after canonical re-score. |
| exp009 | rule_redesign | top4-line-colors | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp010 | rule_redesign | top4-line-colors-split-input | passes_local | 13.295454675088038 | 120959 | 162 | 0.790952872326 | promoted | Auto promoted after canonical re-score. |
| exp011 | impl_opt | share-shift-pads | passes_local | 13.296297162954836 | 120959 | 60 | 0.000842487866798 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | drop-unused-sum-axes | passes_local | 13.295504213554668 | 120959 | 156 | -0.000792949400168 | not_better | Passed but did not improve local_points. |
| exp013 | impl_opt | uniform-line-color-where | passes_local | 13.297289234729917 | 120839 | 60 | 0.00099207177508 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | cast-valid-area | passes_local | 13.297297506131134 | 120839 | 59 | 8.27140121729e-06 | promoted | Auto promoted after canonical re-score. |
| exp015 | rule_redesign | canonical-row-orient | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp016 | rule_redesign | canonical-row-orient-v2 | passes_local | 13.630217687257748 | 86612 | 51 | 0.332920181127 | promoted | Auto promoted after canonical re-score. |
| exp017 | impl_opt | conv-color-map | passes_local | 13.672524981698361 | 83012 | 61 | 0.0423072944406 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
