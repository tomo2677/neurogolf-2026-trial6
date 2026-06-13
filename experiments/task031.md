# task031 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.377825405182378 | 1983 | 60 | 2026-06-13T14:00:25+09:00 | exp026 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | passes_local | 13.77722632138248 | 72992 | 1823 | 1.30098876995 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | int32-index-path | passes_local | 14.169637922050072 | 48708 | 1824 | 0.392411600668 | promoted | Auto promoted after canonical re-score. |
| exp003 | rule_redesign | window12-crop | passes_local | 15.395322877703496 | 14508 | 326 | 1.22568495565 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | mask-fgcolor-crop | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp005 | impl_opt | mask-fgcolor-crop-zero | passes_local | 15.83717061069487 | 9205 | 331 | 0.441847732991 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | remove-unused-zero-i64 | passes_local | 15.83727548196548 | 9205 | 330 | 0.000104871270612 | promoted | Auto promoted after canonical re-score. |
| exp007 | rule_redesign | output7x8-crop | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp008 | rule_redesign | output7x8-crop-v2 | passes_local | 16.338706464610006 | 5333 | 442 | 0.501430982645 | promoted | Auto promoted after canonical re-score. |
| exp009 | impl_opt | gather-1d-crop | passes_local | 16.33905284493907 | 5333 | 440 | 0.000346380329063 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | gather-1d-no-flat-index | passes_local | 16.390045328502453 | 5053 | 433 | 0.0509924835634 | promoted | Auto promoted after canonical re-score. |
| exp011 | impl_opt | full-rank-ch0-slice | passes_local | 16.39022762729067 | 5053 | 432 | 0.000182298788218 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | broadcast-coordinate-grids | passes_local | 16.616338201208286 | 4304 | 71 | 0.226110573918 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | int32-crop-gather | passes_local | 16.72436894542199 | 3856 | 71 | 0.108030744214 | promoted | Auto promoted after canonical re-score. |
| exp015 | impl_opt | fixed-width12-input-valid | passes_local | 16.8038388607171 | 3568 | 59 | 0.0794699152951 | promoted | Auto promoted after canonical re-score. |
| exp016 | impl_opt | axis-gather-crop | passes_local | 17.071233678373304 | 2719 | 57 | 0.267394817656 | promoted | Auto promoted after canonical re-score. |
| exp017 | impl_opt | bool-axis-gather-crop | passes_local | 17.091612840709956 | 2663 | 57 | 0.0203791623367 | promoted | Auto promoted after canonical re-score. |
| exp018 | impl_opt | cast-before-squeeze | passes_local | 17.097512562837146 | 2647 | 57 | 0.00589972212719 | promoted | Auto promoted after canonical re-score. |
| exp019 | rule_redesign | bg-height | passes_local | 17.12450070755479 | 2575 | 57 | 0.0269881447176 | promoted | Auto promoted after canonical re-score. |
| exp021 | impl_opt | bool-crop-pad-opset13 | passes_local | 17.260205541591297 | 2235 | 63 | 0.135704834037 | promoted | Auto promoted after canonical re-score. |
| exp022 | impl_opt | default-false-pad | passes_local | 17.260640797310902 | 2235 | 62 | 0.000435255719605 | promoted | Auto promoted after canonical re-score. |
| exp024 | impl_opt | less-valid-nonzero | passes_local | 17.303787360653594 | 2139 | 61 | 0.0431465633427 | promoted | Auto promoted after canonical re-score. |
| exp026 | rule_redesign | input11-window | passes_local | 17.377825405182378 | 1983 | 60 | 0.0740380445288 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
