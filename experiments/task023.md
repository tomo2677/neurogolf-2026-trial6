# task023 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 14.962243826658838 | 22779 | 95 | 2026-06-13T18:53:40+09:00 | exp030 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | build_failed |  |  |  |  | build_failed | Candidate did not build. |
| exp002 | impl_opt | h002 | passes_local | 10.087275463881577 | 2988096 | 7712 | 0.685827600454 | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | reduce-steps-10 | passes_local | 10.267913950709392 | 2494280 | 6432 | 0.180638486828 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | reduce-steps-9 | passes_local | 10.372153991401705 | 2247372 | 5792 | 0.104240040692 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | reduce-steps-8 | passes_local | 10.488538196543056 | 2000464 | 5152 | 0.116384205141 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | reduce-steps-7 | passes_local | 10.62027396320226 | 1753556 | 4512 | 0.131735766659 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | reduce-steps-6 | passes_local | 10.77203547963 | 1506648 | 3872 | 0.151761516428 | promoted | Auto promoted after canonical re-score. |
| exp008 | impl_opt | reduce-steps-5 | passes_local | 10.951021768350964 | 1259740 | 3232 | 0.178986288721 | promoted | Auto promoted after canonical re-score. |
| exp010 | impl_opt | reduce-steps-4 | fails_local | 0.0 | 1012832 | 2592 | -10.9510217684 | fails_local | Candidate did not pass local validation. |
| exp011 | impl_opt | step4-remaining-square | passes_local | 11.16476132731946 | 1017332 | 2592 | 0.213739558968 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | pad18-shift-axes | passes_local | 11.16601514756348 | 1017332 | 1314 | 0.00125382024402 | promoted | Auto promoted after canonical re-score. |
| exp013 | impl_opt | bool-internal | passes_local | 11.75642834866018 | 563116 | 1314 | 0.590413201097 | promoted | Auto promoted after canonical re-score. |
| exp014 | rule_redesign | window11-cover | passes_local | 13.750662547972082 | 75507 | 1322 | 1.99423419931 | promoted | Auto promoted after canonical re-score. |
| exp015 | rule_redesign | window9x11-cover | passes_local | 13.947413022599758 | 61785 | 1322 | 0.196750474628 | promoted | Auto promoted after canonical re-score. |
| exp016 | impl_opt | dedupe-initializers | passes_local | 13.966869947697983 | 61785 | 106 | 0.0194569250982 | promoted | Auto promoted after canonical re-score. |
| exp017 | impl_opt | u8-count-current | passes_local | 14.09646897552834 | 54261 | 107 | 0.12959902783 | promoted | Auto promoted after canonical re-score. |
| exp018 | impl_opt | u8-count-drop-unused-one | passes_local | 14.096487368869953 | 54261 | 106 | 1.83933416125e-05 | promoted | Auto promoted after canonical re-score. |
| exp019 | impl_opt | remove-seed-identities | passes_local | 14.103797855746665 | 53865 | 106 | 0.00731048687671 | promoted | Auto promoted after canonical re-score. |
| exp020 | impl_opt | single-pad-shifts | passes_local | 14.411120259276993 | 39609 | 82 | 0.30732240353 | promoted | Auto promoted after canonical re-score. |
| exp022 | impl_opt | final-pad-axes | passes_local | 14.41122104286951 | 39609 | 78 | 0.000100783592517 | promoted | Auto promoted after canonical re-score. |
| exp023 | impl_opt | cast-zero-f16 | passes_local | 14.411246240354803 | 39609 | 77 | 2.51974852929e-05 | promoted | Auto promoted after canonical re-score. |
| exp030 | impl_opt | conv3-cross-repair | passes_local | 14.962243826658838 | 22779 | 95 | 0.550997586304 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
