"""Registration tests for Qwen3.5-4B (LLM). No torch needed."""

import pytest

from backend import backends
from backend.backends import llm_progress_name


def test_llm_progress_names_match_configs():
    for cfg in backends.get_llm_model_configs():
        assert llm_progress_name(cfg.model_size) == cfg.model_name


def test_qwen35_registered_on_pytorch(monkeypatch):
    monkeypatch.setattr(backends, "get_backend_type", lambda: "pytorch")
    cfgs = {c.model_name: c for c in backends.get_llm_model_configs()}
    assert cfgs["qwen3.5-4b"].hf_repo_id == "Qwen/Qwen3.5-4B"
    assert cfgs["qwen3.5-4b"].model_size == "3.5-4B"
    for size in ("0.8b", "2b"):
        assert cfgs[f"qwen3.5-{size}"].hf_repo_id == f"Qwen/Qwen3.5-{size.upper()}"


def test_qwen35_not_offered_on_mlx(monkeypatch):
    monkeypatch.setattr(backends, "get_backend_type", lambda: "mlx")
    assert "qwen3.5-4b" not in {c.model_name for c in backends.get_llm_model_configs()}


def test_llm_size_pattern_accepts_35():
    from backend import models

    req = models.CaptureSettingsUpdate(llm_model="3.5-4B")
    assert req.llm_model == "3.5-4B"
    for size in ("3.5-0.8B", "3.5-2B"):
        assert models.CaptureSettingsUpdate(llm_model=size).llm_model == size
    with pytest.raises(Exception):
        models.CaptureSettingsUpdate(llm_model="3.5-9B")
