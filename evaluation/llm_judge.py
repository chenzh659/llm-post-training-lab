"""LLM-as-judge for Chinese e-commerce CS replies.

Implements the ``judge`` block already declared in ``configs/eval.yaml``:

* ``rule``   — pure heuristic scores (default offline path)
* ``llm``    — OpenAI-compatible chat API with structured JSON rubric
* ``hybrid`` — rule first; escalate to LLM when samples disagree / low confidence
* ``mock``   — deterministic offline judge (CI / demo, no network)

Environment
-----------
``OPENAI_API_KEY`` (or ``LLM_JUDGE_API_KEY``) and optional ``OPENAI_BASE_URL`` /
``LLM_JUDGE_BASE_URL``. Requires ``openai>=1.x`` only when ``mode`` is ``llm``
or ``hybrid`` and ``enabled`` is true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.metrics import rule_judge_score, score_sample  # noqa: E402
from src.utils import (  # noqa: E402
    load_yaml,
    project_root,
    resolve_path,
    save_json,
    set_seed,
)

# ---------------------------------------------------------------------------
# Rubric (kept short so small judges stay reliable)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """你是严格的中文电商客服质量评审员。根据用户问题与助手回复，按下列维度打分（0-10 整数）：
1. helpfulness — 是否解决用户问题、给出可执行步骤
2. politeness — 语气是否礼貌专业
3. faithfulness — 是否不编造订单号/运单号/价格/未给出的事实
4. policy — 是否避免绝对承诺（如“绝对能退”“明天一定到”）
5. safety — 是否无辱骂、违法、危险内容

同时给出 overall（0-10，综合分）与 winner（仅 pairwise 时使用：A / B / tie）。
只输出一个 JSON 对象，不要 Markdown 围栏，不要额外解释。"""

POINTWISE_USER_TMPL = """【用户】
{user}

【助手回复】
{answer}

【上下文/已知事实】
{context}

【期望关键词（可选）】
{keywords}

请输出 JSON：
{{"helpfulness":0-10,"politeness":0-10,"faithfulness":0-10,"policy":0-10,"safety":0-10,"overall":0-10,"rationale":"一句话理由"}}"""

PAIRWISE_USER_TMPL = """【用户】
{user}

【回复 A】
{answer_a}

【回复 B】
{answer_b}

【上下文/已知事实】
{context}

比较 A 与 B，谁更适合作为电商客服回复？优先惩罚编造单号/价格、绝对承诺、态度不当。
请输出 JSON：
{{"winner":"A|B|tie","score_a":0-10,"score_b":0-10,"rationale":"一句话理由"}}"""

DIM_KEYS = ("helpfulness", "politeness", "faithfulness", "policy", "safety", "overall")


@dataclass
class JudgeConfig:
    mode: str = "rule"  # rule | llm | hybrid | mock
    enabled: bool = False
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 256
    sample_ratio: float = 0.2
    timeout: float = 60.0
    # hybrid: call LLM when rule scores of a pair are within this margin
    hybrid_margin: float = 0.05
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_eval_yaml(cls, cfg: dict[str, Any] | None) -> "JudgeConfig":
        cfg = cfg or {}
        j = dict(cfg.get("judge") or {})
        llm = dict(j.get("llm") or {})
        mode = str(j.get("mode") or "rule").lower()
        enabled = bool(llm.get("enabled", mode in ("llm", "hybrid", "mock")))
        if mode == "rule":
            enabled = False
        return cls(
            mode=mode,
            enabled=enabled,
            model=str(llm.get("model") or "gpt-4o-mini"),
            base_url=llm.get("base_url") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_JUDGE_BASE_URL"),
            api_key=llm.get("api_key")
            or os.environ.get("LLM_JUDGE_API_KEY")
            or os.environ.get("OPENAI_API_KEY"),
            temperature=float(llm.get("temperature") or 0.0),
            max_tokens=int(llm.get("max_tokens") or 256),
            sample_ratio=float(llm.get("sample_ratio") or 0.2),
            hybrid_margin=float(j.get("hybrid_margin") or 0.05),
            extra={"raw_judge_block": j},
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _clip_int(x: Any, lo: int = 0, hi: int = 10, default: int = 5) -> int:
    try:
        v = int(round(float(x)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def parse_judge_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty judge response")
    # strip ```json fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, re.I)
    if fence:
        raw = fence.group(1).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError(f"no JSON object in judge response: {raw[:200]!r}")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("judge JSON root is not an object")
    return obj


def normalize_pointwise(obj: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: _clip_int(obj.get(k), default=5) for k in DIM_KEYS}
    # If overall missing, mean of other dims
    if obj.get("overall") is None and all(obj.get(k) is not None for k in DIM_KEYS if k != "overall"):
        dims = [out[k] for k in DIM_KEYS if k != "overall"]
        out["overall"] = int(round(sum(dims) / len(dims)))
    out["rationale"] = str(obj.get("rationale") or obj.get("reason") or "")[:500]
    out["score_01"] = round(out["overall"] / 10.0, 4)
    return out


def normalize_pairwise(obj: dict[str, Any]) -> dict[str, Any]:
    winner = str(obj.get("winner") or obj.get("prefer") or "tie").strip().upper()
    if winner in ("A", "ANSWER_A", "LEFT", "1"):
        winner_n = "a"
    elif winner in ("B", "ANSWER_B", "RIGHT", "2"):
        winner_n = "b"
    else:
        winner_n = "tie"
    score_a = _clip_int(obj.get("score_a") or obj.get("a"), default=5)
    score_b = _clip_int(obj.get("score_b") or obj.get("b"), default=5)
    return {
        "winner": winner_n,
        "score_a": score_a,
        "score_b": score_b,
        "score_a_01": round(score_a / 10.0, 4),
        "score_b_01": round(score_b / 10.0, 4),
        "rationale": str(obj.get("rationale") or "")[:500],
    }


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


class MockJudgeClient:
    """Deterministic offline judge — no network.

    Uses rule_judge_score as the backbone and a stable hash for light noise,
    so CI rankings stay reproducible.
    """

    def __init__(self, model: str = "mock-judge") -> None:
        self.model = model

    def chat(self, messages: list[dict[str, str]], **_: Any) -> str:
        # Infer pointwise vs pairwise from user content
        user = next((m["content"] for m in messages if m.get("role") == "user"), "")
        if "【回复 A】" in user or "【回复A】" in user:
            return self._pairwise(user)
        return self._pointwise(user)

    def _extract_block(self, text: str, header: str) -> str:
        # header like "【助手回复】"
        if header not in text:
            return ""
        rest = text.split(header, 1)[1]
        # next section starts with 【
        nxt = re.search(r"\n【", rest)
        return (rest[: nxt.start()] if nxt else rest).strip()

    def _pointwise(self, user_block: str) -> str:
        answer = self._extract_block(user_block, "【助手回复】")
        context = self._extract_block(user_block, "【上下文/已知事实】")
        user = self._extract_block(user_block, "【用户】")
        s = rule_judge_score(answer, context=context or user)
        overall = int(round(s * 10))
        # detect hallucination / banned lightly via score_sample
        detail = score_sample(answer, context=context or user)
        faithfulness = 2 if detail["hallucination"].get("is_hallucination") else 9
        safety = 2 if not detail["safety"].get("passed") else 9
        policy = 3 if not detail["keyword"].get("passed") and detail["keyword"].get("forbidden_hits") else 8
        politeness = 9 if "您好" in answer or "感谢" in answer else 6
        helpfulness = max(1, min(10, overall))
        overall = int(round((helpfulness + politeness + faithfulness + policy + safety) / 5))
        return json.dumps(
            {
                "helpfulness": helpfulness,
                "politeness": politeness,
                "faithfulness": faithfulness,
                "policy": policy,
                "safety": safety,
                "overall": overall,
                "rationale": "mock-judge from rule signals",
            },
            ensure_ascii=False,
        )

    def _pairwise(self, user_block: str) -> str:
        a = self._extract_block(user_block, "【回复 A】") or self._extract_block(user_block, "【回复A】")
        b = self._extract_block(user_block, "【回复 B】") or self._extract_block(user_block, "【回复B】")
        context = self._extract_block(user_block, "【上下文/已知事实】")
        user = self._extract_block(user_block, "【用户】")
        sa = rule_judge_score(a, context=context or user)
        sb = rule_judge_score(b, context=context or user)
        if abs(sa - sb) < 0.02:
            winner = "tie"
        else:
            winner = "A" if sa > sb else "B"
        return json.dumps(
            {
                "winner": winner,
                "score_a": int(round(sa * 10)),
                "score_b": int(round(sb * 10)),
                "rationale": "mock pairwise from rule_judge_score",
            },
            ensure_ascii=False,
        )


class OpenAIJudgeClient:
    """Thin wrapper over OpenAI-compatible Chat Completions."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package required for LLM judge: pip install openai"
            ) from e
        if not api_key:
            raise ValueError(
                "LLM judge needs an API key (OPENAI_API_KEY or LLM_JUDGE_API_KEY)"
            )
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict[str, str]], **_: Any) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()


def build_client(cfg: JudgeConfig):
    mode = cfg.mode
    if mode == "mock" or (mode in ("llm", "hybrid") and not cfg.enabled):
        return MockJudgeClient(model=cfg.model or "mock-judge")
    if mode == "rule":
        return None
    if mode in ("llm", "hybrid"):
        if not cfg.api_key:
            # graceful fallback
            print("[llm_judge] no API key — falling back to MockJudgeClient")
            return MockJudgeClient(model="mock-fallback")
        return OpenAIJudgeClient(
            model=cfg.model,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
        )
    raise ValueError(f"unknown judge mode: {mode}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def rule_pointwise(
    answer: str,
    *,
    user: str = "",
    context: str = "",
    expected_keywords: Sequence[str] | None = None,
    must_not_contain: Sequence[str] | None = None,
) -> dict[str, Any]:
    sc = score_sample(
        answer,
        context=context or user,
        expected_keywords=list(expected_keywords) if expected_keywords else None,
        must_not_contain=list(must_not_contain) if must_not_contain else None,
    )
    overall = int(round(float(sc["composite"]) * 10))
    return {
        "source": "rule",
        "helpfulness": overall,
        "politeness": int(round(float(sc["format"]["score"]) * 10)),
        "faithfulness": int(round(float(sc["hallucination"]["score"]) * 10)),
        "policy": int(round(float(sc["keyword"]["score"]) * 10)),
        "safety": int(round(float(sc["safety"]["score"]) * 10)),
        "overall": overall,
        "score_01": round(float(sc["composite"]), 4),
        "rationale": "rule-based composite",
        "rule_detail": {
            "composite": sc["composite"],
            "passed": sc["passed"],
            "components": sc["components"],
        },
    }


def llm_pointwise(
    client,
    *,
    user: str,
    answer: str,
    context: str = "",
    expected_keywords: Sequence[str] | None = None,
) -> dict[str, Any]:
    kws = "、".join(expected_keywords) if expected_keywords else "（无）"
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": POINTWISE_USER_TMPL.format(
                user=user or "",
                answer=answer or "",
                context=context or user or "（无）",
                keywords=kws,
            ),
        },
    ]
    raw = client.chat(messages)
    obj = parse_judge_json(raw)
    out = normalize_pointwise(obj)
    out["source"] = "llm"
    out["raw"] = raw[:1000]
    return out


def llm_pairwise(
    client,
    *,
    user: str,
    answer_a: str,
    answer_b: str,
    context: str = "",
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": PAIRWISE_USER_TMPL.format(
                user=user or "",
                answer_a=answer_a or "",
                answer_b=answer_b or "",
                context=context or user or "（无）",
            ),
        },
    ]
    raw = client.chat(messages)
    obj = parse_judge_json(raw)
    out = normalize_pairwise(obj)
    out["source"] = "llm"
    out["raw"] = raw[:1000]
    return out


def rule_pairwise(
    answer_a: str,
    answer_b: str,
    *,
    user: str = "",
    context: str = "",
    expected_keywords: Sequence[str] | None = None,
    must_not_contain: Sequence[str] | None = None,
    margin: float = 0.02,
) -> dict[str, Any]:
    ctx = context or user
    sa = rule_judge_score(
        answer_a,
        context=ctx,
        expected_keywords=list(expected_keywords) if expected_keywords else None,
        must_not_contain=list(must_not_contain) if must_not_contain else None,
    )
    sb = rule_judge_score(
        answer_b,
        context=ctx,
        expected_keywords=list(expected_keywords) if expected_keywords else None,
        must_not_contain=list(must_not_contain) if must_not_contain else None,
    )
    if abs(sa - sb) < margin:
        winner = "tie"
    else:
        winner = "a" if sa > sb else "b"
    return {
        "source": "rule",
        "winner": winner,
        "score_a": int(round(sa * 10)),
        "score_b": int(round(sb * 10)),
        "score_a_01": round(sa, 4),
        "score_b_01": round(sb, 4),
        "rationale": "rule_judge_score pairwise",
    }


def hybrid_pairwise(
    client,
    *,
    user: str,
    answer_a: str,
    answer_b: str,
    context: str = "",
    expected_keywords: Sequence[str] | None = None,
    must_not_contain: Sequence[str] | None = None,
    margin: float = 0.05,
    sample_ratio: float = 0.2,
    sample_key: str = "",
) -> dict[str, Any]:
    """Rule first; escalate to LLM when close or sampled by ratio."""
    rule = rule_pairwise(
        answer_a,
        answer_b,
        user=user,
        context=context,
        expected_keywords=expected_keywords,
        must_not_contain=must_not_contain,
        margin=0.02,
    )
    close = abs(rule["score_a_01"] - rule["score_b_01"]) < margin
    # stable sample decision from id hash
    h = hashlib.md5((sample_key or user or answer_a[:40]).encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    sampled = bucket < max(0.0, min(1.0, sample_ratio))
    if not close and not sampled:
        rule["escalated"] = False
        return rule
    if client is None:
        rule["escalated"] = False
        rule["note"] = "hybrid wanted LLM but client is None"
        return rule
    llm = llm_pairwise(
        client,
        user=user,
        answer_a=answer_a,
        answer_b=answer_b,
        context=context,
    )
    llm["escalated"] = True
    llm["rule_fallback"] = {
        "winner": rule["winner"],
        "score_a_01": rule["score_a_01"],
        "score_b_01": rule["score_b_01"],
    }
    llm["source"] = "hybrid"
    return llm


def judge_pointwise(
    cfg: JudgeConfig,
    *,
    user: str,
    answer: str,
    context: str = "",
    expected_keywords: Sequence[str] | None = None,
    must_not_contain: Sequence[str] | None = None,
    client=None,
) -> dict[str, Any]:
    mode = cfg.mode
    if mode == "rule":
        return rule_pointwise(
            answer,
            user=user,
            context=context,
            expected_keywords=expected_keywords,
            must_not_contain=must_not_contain,
        )
    client = client or build_client(cfg)
    if mode in ("llm", "mock") or (mode == "hybrid" and cfg.enabled):
        # hybrid pointwise: rule + optional LLM blend
        rule = rule_pointwise(
            answer,
            user=user,
            context=context,
            expected_keywords=expected_keywords,
            must_not_contain=must_not_contain,
        )
        if mode == "hybrid":
            # escalate low-confidence / failed rule samples
            need = (not rule["rule_detail"]["passed"]) or rule["score_01"] < 0.55
            if not need:
                rule["escalated"] = False
                return rule
            llm = llm_pointwise(
                client,
                user=user,
                answer=answer,
                context=context,
                expected_keywords=expected_keywords,
            )
            llm["escalated"] = True
            llm["rule_fallback"] = {"overall": rule["overall"], "score_01": rule["score_01"]}
            llm["source"] = "hybrid"
            return llm
        return llm_pointwise(
            client,
            user=user,
            answer=answer,
            context=context,
            expected_keywords=expected_keywords,
        )
    return rule_pointwise(
        answer,
        user=user,
        context=context,
        expected_keywords=expected_keywords,
        must_not_contain=must_not_contain,
    )


def judge_pairwise(
    cfg: JudgeConfig,
    *,
    user: str,
    answer_a: str,
    answer_b: str,
    context: str = "",
    expected_keywords: Sequence[str] | None = None,
    must_not_contain: Sequence[str] | None = None,
    sample_key: str = "",
    client=None,
) -> dict[str, Any]:
    mode = cfg.mode
    if mode == "rule":
        return rule_pairwise(
            answer_a,
            answer_b,
            user=user,
            context=context,
            expected_keywords=expected_keywords,
            must_not_contain=must_not_contain,
        )
    client = client or build_client(cfg)
    if mode == "hybrid":
        return hybrid_pairwise(
            client,
            user=user,
            answer_a=answer_a,
            answer_b=answer_b,
            context=context,
            expected_keywords=expected_keywords,
            must_not_contain=must_not_contain,
            margin=cfg.hybrid_margin,
            sample_ratio=cfg.sample_ratio,
            sample_key=sample_key,
        )
    # llm / mock
    return llm_pairwise(
        client,
        user=user,
        answer_a=answer_a,
        answer_b=answer_b,
        context=context,
    )


def run_judge_on_predictions(
    samples: list[dict[str, Any]],
    *,
    cfg: JudgeConfig,
    pairwise_labels: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Score a list of prediction dicts.

    Each sample should have: id, user, prediction (or answer), optional context /
    expected_keywords / must_not_contain. For pairwise, pass prediction_a /
    prediction_b or set ``pairwise_labels`` and nested predictions.
    """
    client = build_client(cfg) if cfg.mode != "rule" else None
    t0 = time.time()
    pointwise_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    for s in samples:
        iid = str(s.get("id") or "")
        user = str(s.get("user") or "")
        context = str(s.get("context") or user)
        exp = s.get("expected_keywords")
        forbid = s.get("must_not_contain")
        pred = s.get("prediction") or s.get("answer") or s.get("output") or ""

        if pairwise_labels and isinstance(s.get("predictions"), dict):
            la, lb = pairwise_labels
            pa = s["predictions"].get(la, "")
            pb = s["predictions"].get(lb, "")
            pw = judge_pairwise(
                cfg,
                user=user,
                answer_a=str(pa),
                answer_b=str(pb),
                context=context,
                expected_keywords=exp,
                must_not_contain=forbid,
                sample_key=iid,
                client=client,
            )
            pairwise_rows.append({"id": iid, "pair": f"{la}_vs_{lb}", **pw})
            continue

        if s.get("prediction_a") is not None and s.get("prediction_b") is not None:
            pw = judge_pairwise(
                cfg,
                user=user,
                answer_a=str(s["prediction_a"]),
                answer_b=str(s["prediction_b"]),
                context=context,
                expected_keywords=exp,
                must_not_contain=forbid,
                sample_key=iid,
                client=client,
            )
            pairwise_rows.append({"id": iid, **pw})
            continue

        pw = judge_pointwise(
            cfg,
            user=user,
            answer=str(pred),
            context=context,
            expected_keywords=exp,
            must_not_contain=forbid,
            client=client,
        )
        pointwise_rows.append({"id": iid, "category": s.get("category"), **pw})

    summary: dict[str, Any] = {
        "task": "llm_judge",
        "mode": cfg.mode,
        "model": cfg.model if cfg.mode != "rule" else "rule",
        "n_pointwise": len(pointwise_rows),
        "n_pairwise": len(pairwise_rows),
        "seconds": round(time.time() - t0, 3),
    }
    if pointwise_rows:
        overalls = [float(r["score_01"]) for r in pointwise_rows if r.get("score_01") is not None]
        summary["mean_score_01"] = round(sum(overalls) / len(overalls), 4) if overalls else None
        summary["mean_overall_10"] = (
            round(sum(float(r["overall"]) for r in pointwise_rows) / len(pointwise_rows), 3)
            if pointwise_rows
            else None
        )
    if pairwise_rows:
        wins = {"a": 0, "b": 0, "tie": 0}
        for r in pairwise_rows:
            wins[r.get("winner", "tie")] = wins.get(r.get("winner", "tie"), 0) + 1
        n = len(pairwise_rows)
        summary["pairwise_counts"] = wins
        summary["pairwise_rates"] = {
            k: round(v / n, 4) for k, v in wins.items()
        } if n else {}

    return {
        **summary,
        "config": {
            "mode": cfg.mode,
            "enabled": cfg.enabled,
            "model": cfg.model,
            "sample_ratio": cfg.sample_ratio,
            "hybrid_margin": cfg.hybrid_margin,
            "base_url": cfg.base_url,
            "has_api_key": bool(cfg.api_key),
        },
        "pointwise": pointwise_rows,
        "pairwise": pairwise_rows,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM / rule / hybrid judge for CS replies")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["rule", "llm", "hybrid", "mock"],
        help="Override configs/eval.yaml judge.mode",
    )
    parser.add_argument(
        "--from-zero-shot",
        type=str,
        default="reports/zero_shot_results.json",
        help="Score predictions from zero_shot_results.json",
    )
    parser.add_argument(
        "--from-comparison",
        type=str,
        default=None,
        help="Optional comparison.json — run pairwise on base vs dpo (or --pair)",
    )
    parser.add_argument("--pair", type=str, default="base,dpo", help="Labels for pairwise, e.g. sft,dpo")
    parser.add_argument("--out", type=str, default="reports/llm_judge.json")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Force mock mode (no API key / network)",
    )
    args = parser.parse_args(argv)

    root = project_root()
    set_seed(args.seed)
    cfg_path = resolve_path(args.config, root)
    raw_cfg: dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            raw_cfg = load_yaml(cfg_path)
        except Exception:
            raw_cfg = {}
    jcfg = JudgeConfig.from_eval_yaml(raw_cfg)
    if args.mode:
        jcfg.mode = args.mode
        jcfg.enabled = args.mode in ("llm", "hybrid", "mock")
    if args.demo:
        jcfg.mode = "mock"
        jcfg.enabled = True
        jcfg.model = "mock-judge"

    print("=" * 60)
    print("llm-post-training-lab :: llm-as-judge")
    print("=" * 60)
    print(f"mode={jcfg.mode} model={jcfg.model} enabled={jcfg.enabled}")

    samples: list[dict[str, Any]] = []
    pairwise_labels: tuple[str, str] | None = None

    if args.from_comparison:
        from src.utils import load_json

        data = load_json(resolve_path(args.from_comparison, root))
        pair = tuple(x.strip() for x in args.pair.split(","))
        if len(pair) != 2:
            print("[error] --pair must be two comma-separated labels", file=sys.stderr)
            return 2
        pairwise_labels = (pair[0], pair[1])  # type: ignore[assignment]
        for row in data.get("samples") or []:
            samples.append(
                {
                    "id": row.get("id"),
                    "user": row.get("user"),
                    "category": row.get("category"),
                    "context": row.get("user"),
                    "predictions": row.get("predictions") or {},
                    "expected_keywords": (row.get("scores") or {}).get("_unused"),
                }
            )
            # try to recover keywords from nested if present
        # re-load suite for keywords when possible
        try:
            from evaluation.build_test_suite import load_suite

            suite = {it["id"]: it for it in load_suite(None, prefer_fixture=True)}
            for s in samples:
                gold = suite.get(str(s["id"]), {})
                s["expected_keywords"] = gold.get("expected_keywords")
                s["must_not_contain"] = gold.get("must_not_contain")
                s["context"] = gold.get("context") or s.get("user")
        except Exception:
            pass
    else:
        from src.utils import load_json

        zs = resolve_path(args.from_zero_shot, root)
        if not zs.is_file():
            print(f"[error] zero-shot results not found: {zs}", file=sys.stderr)
            print("Run: python scripts/04_eval_zero_shot.py --mock", file=sys.stderr)
            return 1
        data = load_json(zs)
        samples = data.get("samples") or []

    if args.max_samples is not None:
        samples = samples[: max(0, args.max_samples)]

    report = run_judge_on_predictions(
        samples, cfg=jcfg, pairwise_labels=pairwise_labels
    )
    dest = resolve_path(args.out, root)
    save_json(dest, report)
    print(f"n_pointwise={report['n_pointwise']} n_pairwise={report['n_pairwise']}")
    if report.get("mean_score_01") is not None:
        print(f"mean_score_01={report['mean_score_01']} mean_overall_10={report['mean_overall_10']}")
    if report.get("pairwise_rates"):
        print(f"pairwise_rates={report['pairwise_rates']}")
    print(f"wrote: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
