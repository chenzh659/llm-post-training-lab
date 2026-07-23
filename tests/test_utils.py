"""Unit tests for src.utils."""

from __future__ import annotations

from pathlib import Path

from src.utils import (
    ensure_dir,
    format_chat,
    get_device,
    load_json,
    project_root,
    read_jsonl,
    resolve_path,
    save_json,
    set_seed,
    write_jsonl,
)


def test_project_root_exists() -> None:
    root = project_root()
    assert root.is_dir()
    assert (root / "README.md").is_file()
    assert (root / "src").is_dir()


def test_set_seed_no_crash() -> None:
    set_seed(42)
    set_seed(0)


def test_get_device_returns_known() -> None:
    dev = get_device()
    assert dev in ("cuda", "mps", "cpu")


def test_json_and_jsonl_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "t.json"
    ensure_dir(p.parent)
    save_json(p, {"a": 1, "中文": "ok"})
    assert load_json(p)["a"] == 1
    assert load_json(p)["中文"] == "ok"

    jl = tmp_path / "x.jsonl"
    write_jsonl(jl, [{"x": 1}, {"x": 2}])
    rows = read_jsonl(jl)
    assert len(rows) == 2
    assert rows[0]["x"] == 1


def test_read_jsonl_missing(tmp_path: Path) -> None:
    assert read_jsonl(tmp_path / "nope.jsonl") == []


def test_format_chat_fallback() -> None:
    text = format_chat(
        [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好"}],
        tokenizer=None,
        add_generation_prompt=True,
    )
    assert "你好" in text
    assert "assistant" in text.lower() or "<|assistant|>" in text


def test_resolve_path_relative() -> None:
    root = project_root()
    p = resolve_path("README.md", root)
    assert p.is_file()
