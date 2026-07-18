# LLM Post-Training Lab

**中文电商智能客服助手 · 端到端后训练实验工程**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Demo Mode](https://img.shields.io/badge/Demo-CPU%20OK-orange.svg)](#quick-start)

面向简历与工程实践的 **LLM Post-Training** 完整流水线：领域数据构建 → SFT → DPO → 规则评测 / 错误分析 → 推理部署与可复现报告。业务场景固定为 **中文电商智能客服助手**。

> 目标不是堆 loss 曲线，而是产出可复现、可对比、可解释的 **业务指标 + 对齐指标 + 推理性能** 闭环。  
> **Loss ≠ Quality** — 详见 [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md)。

---

## 项目目标

| 维度 | 说明 |
|------|------|
| **业务** | 中文电商客服：礼貌、准确、可行动；拒答越权与不编造单号/价格 |
| **训练** | SFT (LoRA/QLoRA) → DPO；配置驱动 |
| **评测** | 格式合规、关键词、幻觉、安全、胜率、错误 taxonomy — **不只看 train loss** |
| **工程** | 脚本一键、`--demo` 无 GPU 可跑、全量 GPU 可复现 |
| **简历** | 数据工程、对齐、评测体系、服务化与实验管理 |

---

## 推荐基座模型

| 场景 | 模型 |
|------|------|
| **Demo / CI** | `Qwen/Qwen2.5-0.5B-Instruct` |
| **质量更好** | `Qwen/Qwen2.5-1.5B-Instruct` |

---

## 目录结构（与仓库一致）

```text
llm-post-training-lab/
├── configs/                 # data.yaml, sft.yaml, dpo.yaml, eval.yaml, deploy.yaml
├── data/
│   ├── raw/                 # sft_raw / preference_raw
│   ├── processed/           # cleaned JSONL
│   ├── splits/              # train/val/test .sft / .pref
│   └── examples/preview.jsonl
├── evaluation/              # metrics, zero_shot, compare, error_analysis, fixtures
├── scripts/
│   ├── 01_build_data.py … 08_bench_serving.py
│   ├── run_pipeline.py      # --stage data|sft|dpo|eval|all  [--demo]
│   └── smoke_test.py        # 无模型下载冒烟
├── src/
│   ├── data/                # generate / clean / analyze / split
│   ├── train/               # sft_train / dpo_train
│   └── utils.py
├── reports/                 # 指标 JSON、FINAL_REPORT.md
├── requirements.txt
├── Makefile                 # make smoke | make demo (optional)
├── LICENSE                  # Apache-2.0
└── README.md
```

---

## 流水线 Stages 1–10

| Stage | 名称 | 入口 |
|------:|------|------|
| 1 | 环境与冒烟 | `python scripts/smoke_test.py` |
| 2–3 | 数据构建 / 清洗切分 | `python scripts/01_build_data.py` |
| 4 | SFT | `python scripts/02_sft_train.py`（或 `--demo`） |
| 5–6 | 偏好 + DPO | `01_build_data` + `03_dpo_train.py` |
| 7 | 离线评测 | `04_eval_zero_shot.py` / `05_eval_compare.py` |
| 8 | 错误分析 | `06_error_analysis.py` |
| 9 | 服务与压测 | `07_deploy_vllm.py` / `08_bench_serving.py`（`--demo` 离线 mock） |
| 10 | 报告 | `reports/FINAL_REPORT.md` |

一键：

```bash
python scripts/run_pipeline.py --stage all --demo
# stages: data | sft | dpo | eval | deploy | all
```

---

## 关键指标（不只看 Loss）

| 类别 | 指标 |
|------|------|
| 任务质量 | 关键词命中、composite score、pass rate |
| 相对质量 | Base/SFT/DPO 规则胜率 |
| 可靠性 | Hallucination rate（编造单号/价格） |
| 安全 | Safety pass / banned phrase |
| 结构化 | Format compliance（问候/结构/长度） |
| 服务 | TTFT、throughput、p95 latency |

---

## Quick Start

### 1. 环境

```bash
cd llm-post-training-lab
python -m venv .venv

# Windows Git Bash
source .venv/Scripts/activate
# Windows cmd: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

> 无 GPU 时仍可装完整 `requirements.txt`；训练阶段会走 mock。纯冒烟只需标准库 + PyYAML（`pip install pyyaml` 即可跑 `smoke_test` 的大部分路径；metrics 不依赖 torch）。

### 2. 冒烟（无模型下载）

```bash
python scripts/smoke_test.py
# 或: make smoke
```

### 3. Demo 全流程（CPU OK）

```bash
python scripts/run_pipeline.py --stage all --demo
# 或: make demo
```

行为：

- 小规模合成数据（约 64 SFT / 32 pref）
- **无 CUDA 时跳过真实训练**，写入 `reports/*_train_metrics.json` 与 `outputs/*/MOCK_CHECKPOINT.txt`
- 评测使用 fixture + mock 生成 / gold 参考，产出 `reports/zero_shot_results.json`、`comparison.json`、`error_analysis.*`

### 4. 分阶段

```bash
python scripts/run_pipeline.py --stage data --demo
python scripts/run_pipeline.py --stage sft --demo
python scripts/run_pipeline.py --stage dpo --demo
python scripts/run_pipeline.py --stage eval --demo
```

### 5. 完整 GPU 路径

```bash
python scripts/01_build_data.py --config configs/data.yaml
python scripts/02_sft_train.py --config configs/sft.yaml
python scripts/03_dpo_train.py --config configs/dpo.yaml
python scripts/04_eval_zero_shot.py --model outputs/sft
python scripts/05_eval_compare.py --base Qwen/Qwen2.5-0.5B-Instruct --sft outputs/sft --dpo outputs/dpo
python scripts/06_error_analysis.py --from-zero-shot reports/zero_shot_results.json
python scripts/07_deploy_vllm.py --config configs/deploy.yaml
python scripts/08_bench_serving.py --config configs/deploy.yaml
```

---

## 样例数据预览

见 [`data/examples/preview.jsonl`](data/examples/preview.jsonl)（5 条多轮客服对话，`messages` 格式）。

---

## 如何复现

1. 固定 `requirements.txt` 与 Python ≥ 3.10  
2. 配置中 `seed: 42`  
3. 记录 `data/splits/split_summary.json` 与 `reports/*`  
4. Demo 命令与 GPU 命令分开写在实验笔记中  

最小 Checklist：

```text
[ ] venv + pip install -r requirements.txt
[ ] python scripts/smoke_test.py          → 全 PASS
[ ] python scripts/run_pipeline.py --stage all --demo
[ ] 打开 reports/FINAL_REPORT.md 与 comparison.json
```

---

## License

**Apache License 2.0** — 见 [`LICENSE`](LICENSE)。

模型权重（如 Qwen2.5）遵循各自模型许可证。

---

## 免责声明

合成/示例数据不代表真实用户隐私；上线前需合规数据、人工审核与风控。
