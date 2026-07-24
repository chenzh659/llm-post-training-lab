#!/usr/bin/env python3
"""Gradio interactive demo — Chinese e-commerce CS chat + offline rule scores.

Modes
-----
* **mock** (default): no model download / GPU; deterministic template replies
* **base / sft / dpo**: load HF base or local LoRA adapters under ``outputs/``

Examples
--------
python scripts/11_gradio_demo.py --mock
python scripts/11_gradio_demo.py --model dpo --share
python scripts/11_gradio_demo.py --model base --base-model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.utils import configure_stdio_utf8

    configure_stdio_utf8()
except Exception:
    pass

from src.demo.chat_engine import (  # noqa: E402
    DEFAULT_BASE,
    DEFAULT_SYSTEM,
    EXAMPLE_PROMPTS,
    ModelBundle,
    format_score_markdown,
    generate_reply,
    history_to_messages,
    load_model_bundle,
    score_reply,
)


def build_ui(
    *,
    default_model: str = "mock",
    base_model: str = DEFAULT_BASE,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    max_new_tokens: int = 256,
) -> Any:
    try:
        import gradio as gr
    except ImportError as e:
        raise SystemExit(
            "gradio is required for the demo UI.\n"
            "  pip install 'gradio>=4.44.0,<6.0.0'\n"
            f"ImportError: {e}"
        ) from e

    state: dict[str, Any] = {
        "bundle": load_model_bundle(
            default_model, base_model=base_model, force_mock=(default_model == "mock")
        ),
        "base_model": base_model,
    }

    def _status_md(bundle: ModelBundle) -> str:
        bits = [
            f"**Model**: `{bundle.name}` · mode=`{bundle.mode}` · device=`{bundle.device}`",
        ]
        if bundle.path:
            bits.append(f"path=`{bundle.path}`")
        if bundle.base_model:
            bits.append(f"base=`{bundle.base_model}`")
        if bundle.load_error:
            bits.append(f"\n\n⚠️ {bundle.load_error}")
        return "  \n".join(bits)

    def on_load_model(choice: str) -> str:
        bundle = load_model_bundle(
            choice,
            base_model=state["base_model"],
            force_mock=(str(choice).lower() == "mock"),
        )
        state["bundle"] = bundle
        return _status_md(bundle)

    def on_chat(
        message: str,
        history: list,
        temperature: float,
        max_tokens: int,
    ):
        message = (message or "").strip()
        empty_summary = {
            "composite": None,
            "passed": False,
            "category": "—",
            "expected_keywords": [],
            "format_score": None,
            "keyword_hit_rate": None,
            "hallucination": False,
            "invented_ids": [],
            "safety_passed": True,
        }
        if not message:
            return history or [], format_score_markdown(empty_summary)

        bundle: ModelBundle = state["bundle"]
        prior = history_to_messages(history)
        reply, latency_ms = generate_reply(
            bundle,
            message,
            history=prior,
            system=DEFAULT_SYSTEM,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
        )
        summary = score_reply(reply, message)
        # Prefer Gradio messages format: [{role, content}, ...]
        new_hist = list(history or [])
        if new_hist and isinstance(new_hist[0], (list, tuple)):
            new_hist = new_hist + [[message, reply]]
        else:
            new_hist = new_hist + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
        return new_hist, format_score_markdown(summary, latency_ms=latency_ms)

    def on_clear():
        empty_score = format_score_markdown(
            {
                "composite": None,
                "passed": False,
                "category": "—",
                "expected_keywords": [],
                "format_score": None,
                "keyword_hit_rate": None,
                "hallucination": False,
                "invented_ids": [],
                "safety_passed": True,
            }
        )
        return [], empty_score

    def fill_example(example: str) -> str:
        # example like "退换货 | 客服您好..."
        if " | " in example:
            return example.split(" | ", 1)[1]
        return example

    example_labels = [f"{c} | {p}" for c, p in EXAMPLE_PROMPTS]

    with gr.Blocks(
        title="电商智能客服 · Post-Training Demo",
        theme=gr.themes.Soft(),
        css="""
        .score-panel { min-height: 220px; }
        footer { visibility: hidden }
        """,
    ) as demo:
        gr.Markdown(
            """
# 中文电商智能客服 · 交互 Demo

**LLM Post-Training Lab** — 对比 Base / SFT / DPO（或 Mock）回复，并实时展示 **规则评测**
（格式 · 关键词 · 幻觉 · 安全 · composite）。

默认 **Mock** 无需 GPU / 权重；有 `outputs/sft` 或 `outputs/dpo` 时可切换真实 adapter。
            """.strip()
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=420,
                    show_copy_button=True,
                    type="messages",
                )
                with gr.Row():
                    msg = gr.Textbox(
                        label="用户消息",
                        placeholder="例如：物流怎么还没更新？单号：ORD-2026-10086",
                        lines=2,
                        scale=4,
                    )
                    send = gr.Button("发送", variant="primary", scale=1)
                with gr.Row():
                    clear_btn = gr.Button("清空对话")
                    examples = gr.Dropdown(
                        label="示例问题",
                        choices=example_labels,
                        value=None,
                        interactive=True,
                    )

            with gr.Column(scale=2):
                model_dd = gr.Dropdown(
                    label="模型",
                    choices=["mock", "base", "sft", "dpo"],
                    value=default_model if default_model in ("mock", "base", "sft", "dpo") else "mock",
                    interactive=True,
                )
                load_btn = gr.Button("加载 / 切换模型")
                status = gr.Markdown(_status_md(state["bundle"]))
                temperature = gr.Slider(0.0, 1.2, value=0.7, step=0.05, label="Temperature")
                max_tokens = gr.Slider(32, 512, value=max_new_tokens, step=16, label="Max new tokens")
                score_md = gr.Markdown(
                    format_score_markdown(
                        {
                            "composite": None,
                            "passed": False,
                            "category": "—",
                            "expected_keywords": [],
                            "format_score": None,
                            "keyword_hit_rate": None,
                            "hallucination": False,
                            "invented_ids": [],
                            "safety_passed": True,
                        }
                    ),
                    elem_classes=["score-panel"],
                )
                gr.Markdown(
                    """
**说明**

| 选项 | 含义 |
|------|------|
| mock | 模板回复，CI / 无 GPU |
| base | `Qwen2.5-*-Instruct` HF 基座 |
| sft / dpo | `outputs/sft` · `outputs/dpo` LoRA |

规则裁判见 `evaluation/metrics.py`；流水线：`scripts/run_pipeline.py`。
                    """.strip()
                )

        load_btn.click(on_load_model, inputs=[model_dd], outputs=[status])
        model_dd.change(on_load_model, inputs=[model_dd], outputs=[status])

        send.click(
            on_chat,
            inputs=[msg, chatbot, temperature, max_tokens],
            outputs=[chatbot, score_md],
        ).then(lambda: "", outputs=[msg])
        msg.submit(
            on_chat,
            inputs=[msg, chatbot, temperature, max_tokens],
            outputs=[chatbot, score_md],
        ).then(lambda: "", outputs=[msg])
        clear_btn.click(on_clear, outputs=[chatbot, score_md])
        examples.change(fill_example, inputs=[examples], outputs=[msg])

        gr.Markdown(
            f"<sub>base_model default = `{base_model}` · server `{server_name}:{server_port}`</sub>"
        )

    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gradio demo for llm-post-training-lab")
    p.add_argument(
        "--model",
        type=str,
        default="mock",
        help="mock | base | sft | dpo | HF id | local path",
    )
    p.add_argument("--base-model", type=str, default=DEFAULT_BASE)
    p.add_argument("--mock", action="store_true", help="Force mock mode (alias for --model mock)")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Gradio public share link")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model = "mock" if args.mock else args.model
    demo = build_ui(
        default_model=model,
        base_model=args.base_model,
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_new_tokens=args.max_new_tokens,
    )
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
