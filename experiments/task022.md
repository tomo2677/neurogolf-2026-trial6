# task022 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 15.988964589858185 | 8102 | 91 | 2026-06-13T09:38:30+09:00 | exp007 |

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
| exp005 | impl_opt | single-pad-shifts | passes_local | 15.894575797973378 | 8912 | 92 | 0.0924386484384 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | pad-onehot3-output | passes_local | 15.988842541893183 | 8102 | 92 | 0.0942667439198 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | remove-unused-invalid | passes_local | 15.988964589858185 | 8102 | 91 | 0.000122047965002 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
