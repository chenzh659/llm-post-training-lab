# Error Analysis Report

- Samples: **20**
- Items with ≥1 error label: **15** (rate 75.00%)
- Model / source: `demo_errors`

## Primary error distribution

| Error | Count | Rate |
|-------|------:|-----:|
| 幻觉 | 0 | 0.00% |
| 政策错误 | 0 | 0.00% |
| 格式违规 | 5 | 25.00% |
| 态度不当 | 10 | 50.00% |
| 答非所问 | 0 | 0.00% |
| 信息缺失 | 0 | 0.00% |

## Multi-label distribution

| Error | Count | Rate |
|-------|------:|-----:|
| 幻觉 | 5 | 25.00% |
| 政策错误 | 10 | 50.00% |
| 格式违规 | 5 | 25.00% |
| 态度不当 | 10 | 50.00% |
| 答非所问 | 0 | 0.00% |
| 信息缺失 | 0 | 0.00% |

## Primary errors by category

- **优惠活动**: 态度不当:1
- **商品咨询**: 格式违规:2
- **投诉建议**: 态度不当:2
- **支付问题**: 态度不当:3, 格式违规:1
- **物流查询**: 格式违规:1, 态度不当:1
- **账户订单**: 格式违规:1
- **退换货**: 态度不当:3

## Representative examples

### 格式违规
- `fix_002` (物流查询)
  - user: 买的无线蓝牙耳机怎么还没发货？物流也不更新。
  - pred: 好的。
  - reasons: ['回复过短/空白或结构不合格']
- `fix_006` (商品咨询)
  - user: 粉色床上四件套材质和尺寸以哪里为准？能保证和详情完全一样吗？
  - pred: 好的。
  - reasons: ['回复过短/空白或结构不合格']

### 态度不当
- `fix_001` (退换货)
  - user: 客服您好，护眼台灯签收3天了，七天无理由怎么退？单号：ORD-2026-88421
  - pred: 滚，自己查去。运单号SF9988776655443，明天一定到。
  - reasons: ["编造标识: ids=['SF9988776655443'] prices=[]", "命中不当/禁用表达: ['滚']", "不当政策承诺/禁语: ['明天一定到']"]
- `fix_004` (支付问题)
  - user: 支付失败怎么办？订单号是A202605288877。
  - pred: 您好。可以退，绝对能退，我保证一定给您退款。祝您购物愉快！
  - reasons: ["命中不当/禁用表达: ['我保证一定', '绝对能退']", "不当政策承诺/禁语: ['绝对能退', '我保证一定']"]

## Taxonomy notes

| Label | Detection heuristic |
|-------|---------------------|
| 幻觉 | Order/tracking IDs or prices in answer not present in context |
| 政策错误 | Absolute refund/discount promises or must_not_contain hits |
| 格式违规 | Empty/too short or failed format_compliance |
| 态度不当 | Banned / abusive / unprofessional phrases |
| 答非所问 | Zero keyword hits and very low overlap with user text |
| 信息缺失 | Partial keyword coverage (<50%) without full off-topic |

