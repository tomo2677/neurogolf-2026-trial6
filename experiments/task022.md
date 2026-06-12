# task022 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 15.80213714953493 | 9752 | 124 | 2026-06-13T07:58:35+09:00 | exp004 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | colorgrid-shifted-gray-slots | passes_local | 14.3302077462862 | 42871 | 165 | 2.21362842047 | promoted | Auto promoted after canonical re-score. |
| exp002 | impl_opt | bool-shift-center-const | passes_local | 14.64798784989526 | 31170 | 150 | 0.317780103609 | promoted | Auto promoted after canonical re-score. |
| exp003 | rule_redesign | window11-gray-overlay | passes_local | 15.79890220939075 | 9752 | 156 | 1.1509143595 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | dedupe-initializers | passes_local | 15.80213714953493 | 9752 | 124 | 0.00323494014418 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
