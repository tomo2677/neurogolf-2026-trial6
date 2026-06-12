# task020 Cost Experiments

## Current Best
| status | local_points | memory_bytes_approx | params | updated_at | source |
| --- | --- | --- | --- | --- | --- |
| passes_local | 15.192527875897559 | 17928 | 241 | 2026-06-13T08:26:07+09:00 | exp014 |

## Active Hypotheses
Keep at most 5 active rows. Use `impl_opt` for implementation/cost changes and `rule_redesign` for rule changes.

| id | mode | hypothesis | status |
| --- | --- | --- | --- |

## Experiment Log
| exp_id | mode | hypothesis_id | status | local_points | memory_bytes_approx | params | delta | decision | takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp001 | impl_opt | h001 | passes_local | 11.032492222096845 | 1161392 | 2765 |  | promoted | Auto promoted after canonical re-score. |
| exp003 | impl_opt | h002 | passes_local | 11.200523300072877 | 981360 | 2734 | 0.168031077976 | promoted | Auto promoted after canonical re-score. |
| exp004 | impl_opt | colorgrid-dynamic-rotate | passes_local | 12.807473693216181 | 195480 | 1829 | 1.60695039314 | promoted | Auto promoted after canonical re-score. |
| exp005 | impl_opt | int32-rotate-indices | passes_local | 13.127318343015258 | 141468 | 1830 | 0.319844649799 | promoted | Auto promoted after canonical re-score. |
| exp006 | impl_opt | nonzero-from-u8 | passes_local | 13.127325321504065 | 141468 | 1829 | 6.97848880726e-06 | promoted | Auto promoted after canonical re-score. |
| exp007 | impl_opt | const-valid10-mask | passes_local | 13.13997327288537 | 138768 | 2728 | 0.0126479513813 | promoted | Auto promoted after canonical re-score. |
| exp011 | impl_opt | ten-by-ten-rotate-opset13 | passes_local | 15.068313793105194 | 20328 | 244 | 1.92834052022 | promoted | Auto promoted after canonical re-score. |
| exp012 | impl_opt | padded-rotate-gather | passes_local | 15.11274346057429 | 19428 | 250 | 0.0444296674691 | promoted | Auto promoted after canonical re-score. |
| exp014 | impl_opt | gather-1d-rotate | passes_local | 15.192527875897559 | 17928 | 241 | 0.0797844153233 | promoted | Auto promoted after canonical re-score. |

## Archived Summary
- None yet.
