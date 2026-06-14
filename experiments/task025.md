# task025 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 14.211980736538221 | 48393 | 44 | 2026-06-14T22:34:48+09:00 | exp021 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |
| conv-color-map | impl_opt | Replace full-grid ArgMax color decoding with a 1x1 FLOAT Conv over one-hot channels to reduce INT64 memory. | promoted |
| threshold01-broadcast-canon-select | impl_opt | expected_delta 0.1-1.0: keep the current row-canonical projection and color-pool matching, but remove the full-grid `has_col` Expand tensors so canonical input/valid/output selection uses scalar/line broadcasting; failure would show ORT materializes equivalent broadcast temporaries or needs explicit shape alignment for these ops. | promoted |

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
| exp019 | rule_redesign | color-pool-extreme-u8 | passes_local | 14.174816350931167 | 50223 | 48 | 0.502291369233 | promoted | Auto promoted after canonical re-score. |
| exp020 | impl_opt | hu-drop-color8-special | fails_local | 0.0 | 42993 | 47 | -14.1748163509 | fails_local | Candidate did not pass local validation. |
| exp021 | impl_opt | threshold01-broadcast-canon-select | passes_local | 14.211980736538221 | 48393 | 44 | 0.0371643856071 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
