---
language:
  - zh
  - en
license: apache-2.0
library_name: peft
tags:
  - text-generation
  - conversational
  - chinese
  - e-commerce
  - customer-service
  - lora
  - dpo
  - peft
base_model: Qwen/Qwen2.5-0.5B-Instruct
pipeline_tag: text-generation
---

# Model Card: 中文电商智能客服助手 (Chinese E-Commerce CS Assistant)

Post-trained **中文电商智能客服** assistant built in this lab via supervised fine-tuning (SFT) with LoRA/QLoRA, followed by Direct Preference Optimization (DPO). This card follows Hugging Face model-card conventions and is intended for portfolio / reproducibility documentation.

> **Status (2026-07-18):** Real QLoRA SFT + DPO on **RTX 4060 Laptop 8GB** (`Qwen2.5-0.5B-Instruct`). Train: `reports/sft_train_metrics.json` / `dpo_train_metrics.json`. Offline rule eval (`mock: false`, n=20): DPO composite **0.866** > SFT **0.860** > Base **0.737**; see `reports/comparison.json`.

---

## Model Details

### Model Description

- **Developed by:** llm-post-training-lab (portfolio project)
- **Model type:** Causal language model (instruction-tuned CS dialogue assistant)
- **Domain:** 中文电商智能客服（退换货 / 物流 / 优惠 / 支付 / 投诉）
- **Language(s):** Primary: Simplified Chinese (`zh`); limited English for product IDs / SKUs / logistics codes
- **License:** Apache-2.0 (adapters inherit base-model license constraints; verify base license before redistribution)
- **Finetuned from model:** `Qwen/Qwen2.5-0.5B-Instruct` (default in `configs/sft.yaml` for 8GB QLoRA); optional upgrade `Qwen/Qwen2.5-1.5B-Instruct` when VRAM allows
- **Adapters:**
  - **SFT LoRA** — low-rank adapters on attention / MLP projections for multi-turn CS style, policy compliance, and tool-aware reply format
  - **DPO LoRA** (or continued LoRA after SFT) — preference-aligned adapters trained on chosen/rejected CS reply pairs
- **Tasks:**
  - Multi-turn e-commerce customer support dialogue (order status, returns/refunds, shipping, product FAQ)
  - Policy-aware answers (refund windows, warranty phrasing — *template policies only*)
  - Structured / template-friendly replies (greeting → clarify → resolve → close)
  - Optional JSON / function-call style slots for order_id, sku, logistics tracking (*if present in training data*)

### Model Sources

- **Repository:** `llm-post-training-lab` (this workspace)
- **Dataset card:** See [`dataset-card.md`](./dataset-card.md) for training / preference data summary
- **Pipeline:** `scripts/run_pipeline.py` (SFT → DPO → eval)
- **Configs:** `configs/` (base model, LoRA ranks, DPO hyperparams — if present)

---

## Intended Use

### Direct Use

- Research / demo of post-training for **Chinese e-commerce CS** assistants
- Offline evaluation of SFT vs SFT+DPO on CS-style metrics (helpfulness, policy adherence, refusal quality)
- Educational reference for LoRA SFT + DPO recipe documentation

### Downstream Use

- Prototype internal CS copilots (human-in-the-loop only)
- Synthetic dialogue generation for further data iteration
- Ablation baseline for preference optimization methods

### Out-of-Scope Use

**Do not use this model for:**

- Fully autonomous production customer service without human review
- Legal, medical, or financial advice beyond generic store policy templates
- Generating deceptive reviews, fake order confirmations, or social-engineering content
- Processing real PII / payment credentials (training and demos should use synthetic IDs)
- High-stakes refund / chargeback decisions without business systems of record
- Open-domain chat outside e-commerce CS (quality and safety not validated)

---

## Training Data

### Summary

Training uses Chinese e-commerce CS dialogue and preference pairs prepared in this lab.

| Stage | Data role | Description |
|-------|-----------|-------------|
| SFT | Instruction / multi-turn dialogues | CS scenarios: pre-sale FAQ, order inquiry, logistics, return/refund, complaint soft-handling |
| DPO | Preference pairs `(prompt, chosen, rejected)` | Preferred replies: clearer, more polite, more policy-aligned; rejected: rude, hallucinated policy, off-topic, over-promise |

**Details, splits, licenses, and synthetic-data notes:** see [`dataset-card.md`](./dataset-card.md).

> If `dataset-card.md` is not yet filled, treat all data as **lab-generated / synthetic or publicly licensed CS-style text** until documented.

### Data Preprocessing (high level)

- Normalize turns to ChatML / messages format compatible with the base instruct model
- Mask loss on user / system turns during SFT (assistant-only)
- Filter empty turns, extreme length, and obvious PII patterns
- For DPO: ensure chosen ≠ rejected; optional length / reward filtering

---

## Training Procedure

Two-stage post-training:

```text
Base instruct model
    → Stage 1: SFT (LoRA)
    → Stage 2: DPO (LoRA on SFT adapter or merged SFT checkpoint)
    → Eval harness
```

### Stage 1 — Supervised Fine-Tuning (SFT + LoRA)

| Item | Value / notes |
|------|----------------|
| Method | Parameter-efficient SFT with LoRA / PEFT |
| Objective | Next-token prediction on assistant tokens |
| Typical LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (confirm in config) |
| Typical rank / alpha | e.g. `r=16`, `lora_alpha=32` — **confirm in training config** |
| Precision | bf16 / fp16 as available |
| Framework | e.g. Hugging Face TRL / transformers + peft |

### Stage 2 — Direct Preference Optimization (DPO)

| Item | Value / notes |
|------|----------------|
| Method | DPO on preference pairs after SFT |
| Reference policy | SFT model (frozen reference) |
| Loss | DPO logistic preference loss (`beta` from config) |
| Adapters | Continue LoRA training or train new DPO LoRA on SFT base |
| Goal | Improve politeness, policy grounding, reduce over-promise / hallucinated commitments |

### Hyperparameters

Exact learning rates, batch sizes, epochs, `beta`, max sequence length, and seed are defined in project configs and should be logged by `scripts/run_pipeline.py`. Do not treat example numbers in this card as authoritative until synced from training logs.

### Speeds, Sizes, Times

| Artifact | Notes |
|----------|--------|
| Base parameters | ~1.5B (default `Qwen2.5-1.5B-Instruct`; demo `0.5B`) |
| Trainable (LoRA) | Typically &lt;1% of base params (rank-dependent) |
| Adapter size on disk | Usually tens–hundreds of MB |
| Wall-clock | **— run `scripts/run_pipeline.py`** (depends on GPU / sample count) |

---

## Evaluation

### Testing Data & Metrics

- **Held-out CS dialogues** and/or preference pairs (see dataset card)
- Suggested metrics (implementations in eval scripts):
  - Automatic: win-rate vs SFT (LLM-as-judge or reward model), BLEU/ROUGE (weak for dialogue), length stats
  - Task: policy adherence score, refusal correctness, slot accuracy (order_id / tracking)
  - Human (optional): helpfulness, politeness, factuality (1–5)

### Results

> **PLACEHOLDER TEMPLATE** — replace after running the pipeline. Values below are **not** measured results.

| Metric | Base instruct | SFT (LoRA) | SFT + DPO | Notes |
|--------|---------------|------------|-----------|--------|
| CS win-rate vs base (LLM-judge) | 50% (ref) | — run `scripts/run_pipeline.py` | — run `scripts/run_pipeline.py` | Pairwise preferred reply |
| Policy adherence (0–1) | — run `scripts/run_pipeline.py` | — run `scripts/run_pipeline.py` | — run `scripts/run_pipeline.py` | Template policy set |
| Hallucinated commitment rate ↓ | — run `scripts/run_pipeline.py` | — run `scripts/run_pipeline.py` | — run `scripts/run_pipeline.py` | Lower is better |
| Avg. response length (chars) | — run `scripts/run_pipeline.py` | — run `scripts/run_pipeline.py` | — run `scripts/run_pipeline.py` | Chinese characters |
| Refusal quality (off-scope) | — run `scripts/run_pipeline.py` | — run `scripts/run_pipeline.py` | — run `scripts/run_pipeline.py` | Should refuse unsafe asks |
| Latency p50 (ms / reply) | — run `scripts/run_pipeline.py` | — run `scripts/run_pipeline.py` | — run `scripts/run_pipeline.py` | Hardware-dependent |

**How to fill this table:**

```bash
python scripts/run_pipeline.py
# or project-specific eval entrypoint documented in README
```

---

## Limitations

- **Domain-narrow:** Tuned for Chinese e-commerce CS; weak on open-domain or non-retail topics
- **Policy templates only:** Store rules in demos may not match any real merchant; model may invent policies if under-trained
- **Hallucination:** May fabricate order status, logistics timelines, or compensation if not grounded in tools / context
- **Context window:** Long multi-turn histories may drop early constraints
- **Dialect / tone:** Standard written Mandarin bias; dialect and code-mixed slang under-covered unless in data
- **English / multilingual:** Not optimized for full bilingual support beyond IDs and short phrases
- **Stale knowledge:** No live catalog / inventory; product facts freeze at training / prompt content
- **Eval placeholders:** Until metrics are filled, quantitative claims are unsupported

## Bias

- May reflect e-commerce corpus stereotypes (gendered product assumptions, urban logistics norms, mainland China platform conventions)
- Preference data can encode annotator taste (overly formal vs casual CS tone)
- Synthetic data may under-represent minority dialects, elderly users, or accessibility needs
- Refund / complaint scenarios may skew toward “merchant-friendly” or “customer-always-right” depending on pair construction

**Mitigations (recommended):** balanced scenario coverage, explicit system policy, human review on sensitive complaints, audit samples across product categories.

## Safety

- **Not a safety-certified product.** No guarantee of jailbreak resistance.
- Should refuse: generating phishing / smishing scripts, bypassing platform fraud checks, doxxing, adult/illegal goods facilitation
- Should escalate (in product design): self-harm, threats, severe fraud — *wire real escalation in the app, not only the model*
- Prefer **tool-grounded** order facts over free-form memory of user orders
- Red-team prompts for: over-refund promises, fake official seals, social-engineering of couriers, leaking other users’ data

Report unsafe outputs via project issue tracker if publishing adapters.

---

## Hardware Notes

| Phase | Suggested hardware | Notes |
|-------|--------------------|--------|
| SFT LoRA/QLoRA (1.5B) | 1× 8–16GB consumer GPU (e.g. RTX 3060+) or cloud T4/L4 | QLoRA 4-bit further reduces VRAM |
| DPO LoRA (1.5B) | Same or slightly higher VRAM (ref + policy) | Gradient checkpointing recommended |
| Inference (adapter) | ≥6–12GB VRAM for 1.5B bf16; less with 4/8-bit | CPU possible but slow; vLLM on Linux+CUDA |
| Full pipeline | Single GPU lab setup sufficient for small–medium datasets | Multi-GPU optional via accelerate |

Exact VRAM footprints: **— run `scripts/run_pipeline.py`** (log peak memory).

**Software stack (typical):** Python 3.10+, PyTorch 2.x, `transformers`, `peft`, `trl`, `datasets`, CUDA matching torch build.

---

## Environmental Impact

Optional; fill after runs:

| Item | Value |
|------|--------|
| Hardware type | — run `scripts/run_pipeline.py` |
| Hours used | — run `scripts/run_pipeline.py` |
| Cloud provider | local / lab |
| Compute region | — |
| Carbon estimate | — |

---

## Citation

If you use this model card, adapters, or training recipe, please cite the project:

```bibtex
@misc{llm_post_training_lab_cs_zh,
  title        = {Chinese E-Commerce Customer Service Assistant (SFT LoRA + DPO)},
  author       = {{llm-post-training-lab}},
  year         = {2026},
  howpublished = {\url{https://github.com/YOUR_USER/llm-post-training-lab}},
  note         = {Model card and post-training recipe for a Chinese e-commerce CS assistant}
}
```

Also cite the **base model** (e.g. Qwen2.5 technical report / model card) and any upstream datasets listed in [`dataset-card.md`](./dataset-card.md).

---

## Model Card Contact

- **Project:** llm-post-training-lab
- **Card version:** 0.1 (template)
- **Last updated:** 2026-07-18

### Card Authors

- Project maintainer(s) of `llm-post-training-lab`

### Framework Versions

| Component | Version note |
|-----------|--------------|
| Transformers / PEFT / TRL | Pin in `requirements.txt` or lockfile after setup |
| Base model revision | Record commit / snapshot id after download |

---

## Appendix: Artifact Layout (expected)

```text
outputs/
  sft_lora/          # SFT adapter + trainer state
  dpo_lora/          # DPO adapter
  eval/              # metrics JSON / tables → paste into Evaluation
scripts/
  run_pipeline.py    # end-to-end SFT → DPO → eval
dataset-card.md      # data documentation
model-card.md        # this file
```

Replace all cells containing `— run scripts/run_pipeline.py` once evaluation artifacts exist.
