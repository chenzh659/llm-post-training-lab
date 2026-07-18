# Error Analysis Report

- Samples: **20**
- Items with ≥1 error label: **0** (rate 0.00%)
- Model / source: `outputs/dpo`

## Primary error distribution

| Error | Count | Rate |
|-------|------:|-----:|
| 幻觉 | 0 | 0.00% |
| 政策错误 | 0 | 0.00% |
| 格式违规 | 0 | 0.00% |
| 态度不当 | 0 | 0.00% |
| 答非所问 | 0 | 0.00% |
| 信息缺失 | 0 | 0.00% |

## Multi-label distribution

| Error | Count | Rate |
|-------|------:|-----:|
| 幻觉 | 0 | 0.00% |
| 政策错误 | 0 | 0.00% |
| 格式违规 | 0 | 0.00% |
| 态度不当 | 0 | 0.00% |
| 答非所问 | 0 | 0.00% |
| 信息缺失 | 0 | 0.00% |

## Representative examples

## Taxonomy notes

| Label | Detection heuristic |
|-------|---------------------|
| 幻觉 | Order/tracking IDs or prices in answer not present in context |
| 政策错误 | Absolute refund/discount promises or must_not_contain hits |
| 格式违规 | Empty/too short or failed format_compliance |
| 态度不当 | Banned / abusive / unprofessional phrases |
| 答非所问 | Zero keyword hits and very low overlap with user text |
| 信息缺失 | Partial keyword coverage (<50%) without full off-topic |

