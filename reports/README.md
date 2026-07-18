# Reports directory

Artifacts produced by the **llm-post-training-lab** evaluation, training, and serving stages. All JSON is UTF-8; Markdown is human-readable summary.

## Evaluation

| File | Producer | Contents |
|------|----------|----------|
| `zero_shot_results.json` | `scripts/04_eval_zero_shot.py` / `evaluation.zero_shot_eval` | Base (or mock) model generations on the test suite; per-sample composite scores (format, keyword, ROUGE-L char, hallucination, safety, MC); summary + per-category aggregates. |
| `comparison.json` | `scripts/05_eval_compare.py` / `evaluation.compare_models` | Side-by-side base vs SFT vs DPO predictions; rule-based win-rate pairwise matrix; ranking by mean composite. |
| `error_analysis.json` | `scripts/06_error_analysis.py` / `evaluation.error_analysis` | Multi-label error taxonomy counts: 幻觉 / 政策错误 / 格式违规 / 态度不当 / 答非所问 / 信息缺失; per-item labels + signals. |
| `error_analysis.md` | same | Markdown tables + representative failure examples for portfolio / README embeds. |

## Data pipeline (existing)

| File | Producer | Contents |
|------|----------|----------|
| `data_cleaning_stats.json` | `src.data.clean` via `scripts/01_build_data.py` | Input/output counts and filter drop reasons for SFT & preference. |
| `data_length_analysis.json` | `src.data.analyze` | Length distributions, category histograms. |

## Serving / performance

| File | Producer | Contents |
|------|----------|----------|
| `bench_serving.json` | `scripts/08_bench_serving.py` | TTFT (mean/p50/p95), end-to-end latency, requests/s, completion tokens/s; backend = `http` (vLLM) or `transformers_local`. |

## Experiment log

| File | Purpose |
|------|---------|
| `experiment_log_template.md` | Fill-in template: GPU mem, wall time, hyperparams, key metrics for each run. Copy to `experiment_log_<date>.md` per experiment. |

## Offline demo without training

```bash
# Uses evaluation/fixtures/sample_test.jsonl (20 handcrafted cases)
python scripts/04_eval_zero_shot.py --mock
python scripts/05_eval_compare.py --mock --sft gold --dpo mock
python scripts/06_error_analysis.py --demo-errors
python scripts/07_deploy_vllm.py                  # print install + launch cmd
python scripts/08_bench_serving.py --transformers-fallback --num-prompts 4
```

## Metric definitions (short)

- **composite**: weighted blend of format / keyword / lexical / hallucination / safety (/ MC when present).
- **hallucination_rate**: fraction of replies inventing order/tracking IDs or prices absent from context.
- **safety_fail_rate**: banned-phrase hits.
- **win-rate**: pairwise rule judge on composite + mild length preference.
- **TTFT**: time to first token (streaming); non-stream ≈ E2E.
- **p95 latency**: 95th percentile end-to-end request time under configured concurrency.
