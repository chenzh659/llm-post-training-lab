# Experiment Log Template

Copy this file to `reports/experiment_log_YYYYMMDD_<tag>.md` for each run. Fill every `TBD` field.

---

## Metadata

| Field | Value |
|-------|-------|
| Experiment ID | TBD |
| Date | YYYY-MM-DD |
| Operator | TBD |
| Git commit / data snapshot | TBD |
| Seed | 42 |
| Mode | demo / full-GPU |

## Hardware

| Field | Value |
|-------|-------|
| GPU model | e.g. RTX 4060 8GB / A10 / none (CPU) |
| GPU memory total (MiB) | TBD |
| Peak allocated (MiB) | TBD (from report `timing.peak_gpu_memory_mb`) |
| CUDA / driver | TBD |
| CPU / RAM | TBD |
| OS | Windows / Linux / WSL2 |

## Software

| Field | Value |
|-------|-------|
| Python | 3.x.x |
| torch | TBD |
| transformers | TBD |
| peft / trl | TBD |
| vLLM (if any) | TBD / N/A |

## Model & data

| Field | Value |
|-------|-------|
| Base model | Qwen/Qwen2.5-0.5B-Instruct (or 1.5B) |
| SFT adapter / merge path | TBD |
| DPO adapter / merge path | TBD |
| Train set | `data/splits/train.sft.jsonl` (n=TBD) |
| Eval / test suite | `evaluation/fixtures/sample_test.jsonl` or `data/splits/test.sft.jsonl` |
| Max samples evaluated | TBD |

## Training (if applicable)

### SFT

| Hyperparam | Value |
|------------|-------|
| Config | `configs/sft.yaml` |
| Epochs / max steps | TBD |
| LR / scheduler | TBD |
| LoRA r / alpha / dropout | TBD |
| Batch × grad accum | TBD |
| Seq length | TBD |
| Wall time | TBD minutes |
| Final train loss | TBD |
| Eval loss | TBD |

### DPO / preference

| Hyperparam | Value |
|------------|-------|
| Config | `configs/dpo.yaml` |
| β / loss type | TBD |
| Epochs / steps | TBD |
| Wall time | TBD minutes |
| Final loss | TBD |

## Offline metrics

Source files: `reports/zero_shot_results.json`, `reports/comparison.json`, `reports/error_analysis.json`.

### Zero-shot (base)

| Metric | Value |
|--------|------:|
| n | TBD |
| mean_composite | TBD |
| pass_rate | TBD |
| hallucination_rate | TBD |
| safety_fail_rate | TBD |
| mc_accuracy (if any) | TBD |

### Comparison ranking

| Rank | Model | mean_composite | pass_rate | hallu_rate |
|-----:|-------|---------------:|----------:|-----------:|
| 1 | TBD | | | |
| 2 | TBD | | | |
| 3 | TBD | | | |

### Pairwise win-rate

| Pair | Win A | Win B | Tie |
|------|------:|------:|----:|
| base vs sft | | | |
| base vs dpo | | | |
| sft vs dpo | | | |

### Error analysis (primary)

| Error | Count | Rate |
|-------|------:|-----:|
| 幻觉 | | |
| 政策错误 | | |
| 格式违规 | | |
| 态度不当 | | |
| 答非所问 | | |
| 信息缺失 | | |
| **overall error_rate** | | |

## Serving benchmark

Source: `reports/bench_serving.json` (server: vLLM / transformers fallback).

| Metric | Value |
|--------|------:|
| Backend | http / transformers_local |
| Concurrency | TBD |
| Num prompts | TBD |
| Max tokens | TBD |
| TTFT mean (s) | TBD |
| TTFT p95 (s) | TBD |
| E2E mean (s) | TBD |
| E2E p95 (s) | TBD |
| Requests / s | TBD |
| Completion tokens / s | TBD |

## Qualitative notes

- Best improved categories after SFT/DPO: TBD
- Remaining failure modes: TBD
- Safety / hallucination observations: TBD
- Next actions: TBD

## Reproduce commands

```bash
python scripts/01_build_data.py
python scripts/02_sft_train.py --config configs/sft.yaml
python scripts/03_dpo_train.py --config configs/dpo.yaml
python scripts/04_eval_zero_shot.py --mock   # or --model <path>
python scripts/05_eval_compare.py --base <base> --sft <sft> --dpo <dpo>
python scripts/06_error_analysis.py --from-zero-shot reports/zero_shot_results.json
python scripts/07_deploy_vllm.py --config configs/deploy.yaml
python scripts/08_bench_serving.py --base-url http://127.0.0.1:8000/v1
```

## Checklist

```text
[ ] Config snapshot copied under reports/experiments/<id>/
[ ] Metrics tables filled from JSON artifacts
[ ] Peak GPU memory recorded
[ ] Wall-clock times recorded
[ ] Known non-determinism noted (sampling, CUDA)
```
