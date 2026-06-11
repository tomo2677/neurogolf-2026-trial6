# task001 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.433688985227537 | 1905 | 27 | 2026-06-12T01:52:12+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp003 | impl_opt | pad-output | passes_local | 15.404193166422342 | 14652 | 51 | 1.3770529879 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | nonzero-channel-output | passes_local | 15.747175016417662 | 10404 | 30 | 0.342981849995 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | slice-attrs-opset9 | passes_local | 15.748709641694862 | 10404 | 14 | 0.0015346252772 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | upsample-mask-opset8 | passes_local | 15.749573904484532 | 10404 | 5 | 0.000864262789669 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | single-color-factor | passes_local | 16.031349096625128 | 7848 | 5 | 0.281775192141 | promoted | Auto promoted after canonical re-score. |
| exp008 | impl_opt | zero-channel-mask | passes_local | 16.063438795306606 | 7600 | 5 | 0.0320896986815 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | broadcast-spatial | passes_local | 16.097136327803778 | 7348 | 5 | 0.0336975324972 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | reshape-attr-opset4 | score_failed |  |  |  |  | score_failed | Candidate did not pass local validation. |
| exp011 | impl_opt | bool-mask | passes_local | 16.118302593530696 | 7195 | 4 | 0.0211662657269 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | bool-output9 | passes_local | 16.451889705949043 | 5153 | 4 | 0.333587112418 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | uint8-output-pad | passes_local | 17.079916800946766 | 2723 | 29 | 0.628027094998 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | split-color-selector | passes_local | 17.082463646056368 | 2724 | 21 | 0.0025468451096 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | bool-padded-output | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp016 | impl_opt | bool-padded-output-split-input | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp017 | impl_opt | bool-padded-output-opset13 | passes_local | 17.42904141683099 | 1914 | 27 | 0.346577770775 | promoted | Auto promoted after canonical re-score. |
| exp018 | impl_opt | pad-default-false | passes_local | 17.429556747942627 | 1914 | 26 | 0.000515331111636 | promoted | Auto promoted after canonical re-score. |
| exp019 | impl_opt | equal-mask | passes_local | 17.433688985227537 | 1905 | 27 | 0.00413223728491 | promoted | Auto promoted after canonical re-score. |
| exp020 | impl_opt | split-empty-output | score_failed |  |  |  |  | score_failed | Candidate did not pass local validation. |
| exp021 | impl_opt | flat-outer-spatial | fails_local | 0.0 | 1833 | 24 | -17.4336889852 | fails_local | Candidate did not pass local validation. |

## Archived Summary
- None yet.
