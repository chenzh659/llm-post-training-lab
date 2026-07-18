# FINAL_REPORT — LLM Post-Training Lab Experiment Report

> **Domain:** 中文电商智能客服助手  
> **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`（真训默认 / demo）· 可选 `Qwen2.5-1.5B-Instruct`  
> **Seed:** 42  
> **Mode:** **Real GPU QLoRA 已完成**（RTX 4060 Laptop 8GB）；离线规则评测图仍可为 demo mock  
> **Date:** 2026-07-18  

---

## 0. 如何解读本报告（必读）

### Loss ≠ Quality

| 误区 | 正确做法 |
|------|----------|
| 只报 train loss 下降 | 必须并列业务指标：关键词命中、幻觉率、安全通过率、格式合规、胜率 |
| 认为 ROUGE/BLEU 高 = 客服好 | 客服更重**可执行性、政策边界、不编造单号**；字面重合次之 |
| 忽略安全与拒答 | 红队/违禁语命中率、绝对承诺（“绝对能退”）必须单独统计 |
| 跨机器比吞吐 | 注明 GPU、量化、并发、max tokens；p95 与 TTFT 比均值更重要 |

**过程量（train loss / DPO loss）** 仅用于确认训练在收敛。  
**决策量** 以 Stage 7–8 的离线评测与错误分析为准。

---

## 1. Stage 1 — 环境与基线

| 项 | 值 |
|----|-----|
| Python | 3.10+（冒烟环境见 `scripts/smoke_test.py` 输出） |
| 核心依赖 | `requirements.txt`（torch / transformers / peft / trl / rich …） |
| 冒烟 | `python scripts/smoke_test.py` — 无模型下载 |
| Demo 一键 | `python scripts/run_pipeline.py --stage all --demo` |

**验收：** 导入 `src.*` / `evaluation.*` 成功；metrics 单测通过。

---

## 2. Stage 2 — 领域数据构建

| 数据集 | 生成量（全量配置） | Demo 量 | 路径 |
|--------|-------------------|---------|------|
| SFT 对话 | ~2000 | 64 | `data/raw/sft_raw.jsonl` |
| Preference 对 | ~800 | 32 | `data/raw/preference_raw.jsonl` |

**场景覆盖：** 商品咨询、物流查询、退换货、优惠活动、投诉建议、账户订单、支付问题。

**样例预览：** 见 `data/examples/preview.jsonl`（5 条）。

**复现：**

```bash
python scripts/01_build_data.py --config configs/data.yaml
# demo
python scripts/01_build_data.py --sft-samples 64 --pref-pairs 32 --skip-plots
```

---

## 3. Stage 3 — 质量过滤与切分

| 步骤 | 脚本 / 模块 | 输出 |
|------|-------------|------|
| 清洗去重 | `src/data/clean.py` | `data/processed/*_clean.jsonl`、`reports/data_cleaning_stats.json` |
| 长度/类目分析 | `src/data/analyze.py` | `reports/data_length_analysis.json` |
| 8:1:1 切分 | `src/data/split.py` | `data/splits/{train,val,test}.{sft,pref}.jsonl` |

### 已有切分摘要（仓库内全量一次跑通后）

| 集 | input | train | val | test |
|----|------:|------:|----:|-----:|
| SFT | 1932 | 1543 | 188 | 201 |
| Preference | 800 | 638 | 78 | 84 |

*Demo 重跑会覆盖为更小规模；以最新 `data/splits/split_summary.json` 为准。*

---

## 4. Stage 4 — SFT 监督微调

| 项 | 全量 GPU（已跑通） | Demo（无 GPU） |
|----|-------------------|----------------|
| 方法 | 4-bit QLoRA SFT（TRL + PEFT） | Mock metrics + 占位 adapter |
| 配置 | `configs/sft.yaml`（0.5B · batch=1 · accum=16 · seq=1024） | `python scripts/02_sft_train.py --demo` |
| 输出 | `outputs/sft/adapter_model.safetensors`（~17MB，gitignored） | `outputs/sft/MOCK_CHECKPOINT.txt` |
| 指标文件 | `reports/sft_train_metrics.json`（`mock: false`） | 同路径（`mock: true`） |

### 训练过程量（Real GPU · RTX 4060 Laptop 8GB · 2026-07-18）

| 指标 | Demo mock（历史） | **Real GPU** |
|------|------------------:|-------------:|
| model | 0.5B（mock） | **Qwen2.5-0.5B-Instruct + QLoRA** |
| train_loss | 1.234 | **0.364** |
| wall_time_sec | ~0.01 | **2989**（~50 min） |
| peak_gpu_memory_mb | null | **1565** |
| num_train_samples | 32 | **1543** |
| epochs / steps | 1 | **2 / 194** |
| mean_token_accuracy | — | **~0.974** |

日志：`reports/sft_train_log.txt`（step loss 3.6 → ~0.07）。

> **解读：** 若 loss 下降但 Stage 7 幻觉率上升，应优先修数据（禁止编造单号）而非继续加 epoch。

---

## 5. Stage 5 — 偏好数据

| 字段 | 说明 |
|------|------|
| prompt / prompt_messages | 用户问题 + system |
| chosen | 礼貌、可执行、政策边界清晰的回复 |
| rejected | 幻觉单号 / 绝对承诺 / 态度差 / 答非所问 等负例类型 |

路径：`data/raw/preference_raw.jsonl` → clean → `data/splits/*.pref.jsonl`。

---

## 6. Stage 6 — DPO 偏好对齐

| 项 | 全量 GPU（已跑通） | Demo（无 GPU） |
|----|-------------------|----------------|
| 方法 | DPO + LoRA on SFT adapter（TRL） | Mock metrics |
| 配置 | `configs/dpo.yaml`（beta=0.1 · max_len=1024） | pipeline `--demo` |
| 输出 | `outputs/dpo/adapter_model.safetensors`（~17MB，gitignored） | mock checkpoint |
| 指标 | `reports/dpo_train_metrics.json`（`mock: false`） | `train_loss≈0.456`（mock） |

### 训练过程量（Real GPU · 接在 SFT 之后）

| 指标 | Demo mock（历史） | **Real GPU** |
|------|------------------:|-------------:|
| base / policy | mock | **0.5B + `outputs/sft` adapter** |
| train_loss | 0.456 | **0.114** |
| wall_time_sec | ~0.01 | **633**（~11 min） |
| peak_gpu_memory_mb | null | **2116** |
| num_train_samples | 32 | **638** |
| epochs / steps | 1 | **1 / 40** |
| 偏好信号 | — | 后期 **rewards/accuracies≈1.0**，margins 上升 |

日志：`reports/dpo_train_log.txt`。

---

## 7. Stage 7 — 离线综合评测

### 7.1 Zero-shot / mock 基线

来源：`reports/zero_shot_results.json`（`--mock` 时为规则模板生成）。

| 指标 | Demo 数值（`--stage eval --demo`，n=20） |
|------|------------------:|
| n | 20 |
| mean_composite | 0.8822 |
| pass_rate | 0.90 |
| hallucination_rate | 0.05 |
| safety_fail_rate | 0.0 |

### 7.2 Base vs SFT vs DPO 对比

来源：`reports/comparison.json`（mock：base 故意偏弱，sft=gold 参考回复）。

| 模型 | mean_composite | pass_rate | hallucination_rate |
|------|---------------:|----------:|-------------------:|
| sft (gold ref) | 0.9575 | 0.80 | 0.15 |
| dpo (mock) | 0.8700 | 0.85 | 0.05 |
| base (weak mock) | 0.6107 | 0.00 | 0.20 |

**成对胜率（规则裁判，示例）：**

| 对局 | win_rate A | win_rate B | tie |
|------|-----------:|-----------:|----:|
| base vs sft | 0.05 | 0.95 | 0.00 |
| base vs dpo | 0.05 | 0.95 | 0.00 |
| sft vs dpo | 0.15 | 0.00 | 0.85 |

> 全量 GPU 实验请替换为真实 checkpoint 路径：  
> `python scripts/05_eval_compare.py --base <hf> --sft outputs/sft --dpo outputs/dpo`

---

## 8. Stage 8 — 错误分析与安全

来源：`reports/error_analysis.json` / `.md`（`--demo-errors` 注入坏回复以展示 taxonomy）。

| 错误类型 | 含义 | 检测要点 |
|----------|------|----------|
| 幻觉 | 编造订单/运单/价格 | 答案有、上下文无 |
| 政策错误 | 绝对退款/额外优惠承诺 | must_not_contain / 绝对用语 |
| 格式违规 | 过短、无结构 | format_compliance |
| 态度不当 | 辱骂/推诿 | banned phrases |
| 答非所问 | 与用户意图无关 | 关键词 0 命中 + 低重合 |
| 信息缺失 | 关键词覆盖不足 | hit_rate < 0.5 |

**Demo 主错误分布（示例）：** 态度不当、格式违规 占比较高（因注入坏样例）；真实模型应以对比实验 JSON 为准。

---

## 9. Stage 9 — 推理服务与压测

| 项 | 说明 |
|----|------|
| vLLM | Linux + CUDA：`python scripts/07_deploy_vllm.py --config configs/deploy.yaml --run` |
| Windows | 打印安装指引；可用 `--fallback-transformers` 演示 API 形态 |
| 压测 | `python scripts/08_bench_serving.py --config configs/deploy.yaml` |
| Demo（无服务） | `python scripts/08_bench_serving.py --demo` → `reports/bench_serving.json` |
| 指标 | TTFT、throughput、p50/p95 latency、peak GPU mem |

### 性能表（Demo mock / 全量后替换）

| 配置 | concurrency | TTFT mean (s) | p95 E2E (s) | tok/s | notes |
|------|------------:|--------------:|------------:|------:|-------|
| demo_mock | 2–4 | ~0.08 | ~0.43 | synthetic | `scripts/08_bench_serving.py --demo` → `reports/bench_serving.json` |
| vLLM 1.5B | 8 | _TBD_ | _TBD_ | _TBD_ | Linux+CUDA |
| transformers fallback | 1 | _TBD_ | _TBD_ | _TBD_ | CPU demo |

Pipeline stage: `python scripts/run_pipeline.py --stage deploy --demo`

---

## 10. Stage 10 — 结论与复现

### 主要结论（Demo）

1. **数据闭环可用：** 合成 SFT + preference → 清洗 → 分层切分，schema 与冒烟测试通过。  
2. **评测不只看 loss：** 规则指标覆盖格式、关键词、幻觉、安全；对比表可区分弱 base 与 gold 参考。  
3. **无 GPU 可演示全链路：** `--demo` 写 mock 训练指标 + fixture/mock 评测，适合简历演示与 CI。  
4. **有 GPU 时：** 去掉 mock，用 `configs/sft.yaml` / `dpo.yaml` 跑真实 QLoRA，再对比真实 adapter。

### 复现命令

```bash
# 环境
python -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate
pip install -r requirements.txt

# 冒烟（无模型）
python scripts/smoke_test.py

# Demo 全流程
python scripts/run_pipeline.py --stage all --demo

# 分阶段
python scripts/run_pipeline.py --stage data --demo
python scripts/run_pipeline.py --stage eval --demo

# 全量数据 + GPU 训练（需 CUDA）
python scripts/01_build_data.py
python scripts/02_sft_train.py
python scripts/03_dpo_train.py
python scripts/04_eval_zero_shot.py --model outputs/dpo
python scripts/05_eval_compare.py --sft outputs/sft --dpo outputs/dpo
python scripts/06_error_analysis.py
```

### 关键产物路径

| 产物 | 路径 |
|------|------|
| 本报告 | `reports/FINAL_REPORT.md` |
| Zero-shot | `reports/zero_shot_results.json` |
| 对比 | `reports/comparison.json` |
| 错误分析 | `reports/error_analysis.md` |
| 数据清洗 | `reports/data_cleaning_stats.json` |
| 预览对话 | `data/examples/preview.jsonl` |

---

## Appendix A — 指标定义摘要

- **composite：** format / keyword / lexical / hallucination / safety /（可选）MC 加权。  
- **pass：** 格式通过 + 安全通过 + 无幻觉 +（有关键词时）关键词通过。  
- **win-rate：** 成对 `rule_judge_score` 比较。  

## Appendix B — 免责声明

合成数据仅供工程演示；不含真实用户隐私。上线前需真实数据、人工抽检与完整风控。模型权重遵循 Qwen 等各自许可证。
