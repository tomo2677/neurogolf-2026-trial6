# task040 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 17.439398837231444 | 1876 | 45 | 2026-06-13T12:40:00+09:00 | exp017 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | rule_redesign | edge-only-guide-marker | passes_local | 16.851843560078375 | 2793 | 664 | 0.693002536024 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | broadcast-edge-masks | passes_local | 17.02168903013228 | 2793 | 124 | 0.169845470054 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | h003 | passes_local | 17.36324788756422 | 1953 | 120 | 0.341558857432 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | top-slice3-pad-axes | passes_local | 17.377825405182378 | 1925 | 118 | 0.0145775176182 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | nonzero-color-idx | passes_local | 17.383716438419615 | 1913 | 118 | 0.00589103323724 | promoted | Auto promoted after canonical re-score. |
| exp011 | impl_opt | top-zero-width2 | passes_local | 17.385687853548 | 1909 | 118 | 0.00197141512838 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | concat-edge-guides | passes_local | 17.408138285110066 | 1889 | 93 | 0.0224504315621 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | concat-replacements | passes_local | 17.418280359874693 | 1889 | 73 | 0.0101420747646 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | bg-scalar-onecell | fails_local | 0.0 | 1881 | 58 | -17.4182803599 | fails_local | Candidate did not pass local validation. |
| exp015 | impl_opt | bg-scalar | passes_local | 17.4259549946278 | 1889 | 58 | 0.00767463475311 | promoted | Auto promoted after canonical re-score. |
| exp016 | impl_opt | orient-corner-equal | passes_local | 17.436799407641928 | 1876 | 50 | 0.0108444130141 | promoted | Auto promoted after canonical re-score. |
| exp017 | impl_opt | slice-chw-axes | passes_local | 17.439398837231444 | 1876 | 45 | 0.00259942958952 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
