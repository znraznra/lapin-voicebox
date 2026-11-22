"""Registration tests for Qwen3.5-4B (LLM) and Qwen Emirati (TTS). No torch needed."""

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


def test_qwen35_not_offered_on_mlx(monkeypatch):
    monkeypatch.setattr(backends, "get_backend_type", lambda: "mlx")
    assert "qwen3.5-4b" not in {c.model_name for c in backends.get_llm_model_configs()}


def test_emirati_config_matches_progress_name():
    cfg = next(c for c in backends.get_tts_model_configs() if c.engine == "qwen_emirati")
    assert cfg.model_name == "qwen-emirati"
    assert cfg.hf_repo_id == "vadimbelsky/qwen3.5-TTS-Emirati"
    assert "qwen_emirati" in backends.TTS_ENGINES


def test_emirati_in_request_patterns():
    from backend import models

    req = models.GenerationRequest(profile_id="p", text="hi", language="ar", engine="qwen_emirati")
    assert req.engine == "qwen_emirati"


def test_llm_size_pattern_accepts_35():
    from backend import models

    req = models.CaptureSettingsUpdate(llm_model="3.5-4B")
    assert req.llm_model == "3.5-4B"
    with pytest.raises(Exception):
        models.CaptureSettingsUpdate(llm_model="3.5-9B")


def test_emirati_preset_voice_ids():
    pytest.importorskip("torch")
    from backend.backends.qwen_emirati_backend import QWEN_EMIRATI_VOICES

    assert [v[0] for v in QWEN_EMIRATI_VOICES] == ["emirati_speaker"]
