# task032 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 18.186555400489105 | 0 | 910 | 2026-06-14T22:21:24+09:00 | ledger |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |
| hu-kernel5-gravity-cap | rule_redesign | expected_delta 0.1-1.0: reduce the column-gravity Conv kernel from 10 rows to 5 rows, cutting about half of the 1010 params if local task sizes never require deeper vertical context; failure would prove generated examples need the current 10-row window. | fails_local |
| threshold01-kernel9 | rule_redesign | expected_delta 0.1-1.0: reduce the column-gravity Conv kernel from 10 rows to 9 rows, saving about 100 params while preserving most vertical context; failure would show the local generator needs the full 10-row window. | promoted |
| threshold01-kernel8 | rule_redesign | expected_delta 0.1-1.0: reduce the column-gravity Conv kernel from 9 rows to 8 rows, saving another 100 params while testing whether one less lower-context row still satisfies local generated examples; failure would show the current 9-row window is the local boundary. | fails_local |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | rule_redesign | hu-kernel5-gravity-cap | fails_local | 0.0 | 0 | 510 | -18.0822943902 | fails_local | Candidate did not pass local validation. |
| exp002 | rule_redesign | threshold01-kernel9 | passes_local | 18.186555400489105 | 0 | 910 | 0.104261010324 | promoted | Auto promoted after canonical re-score. |
| exp003 | rule_redesign | threshold01-kernel8 | fails_local | 0.0 | 0 | 810 | -18.1865554005 | fails_local | Candidate did not pass local validation. |

## Archived Summary
- None yet.
