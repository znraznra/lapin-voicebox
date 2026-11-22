"""VoxCPM2 engine: registration, XPU device patch, and generate() wiring.

The model itself is faked; the XPU patch test uses the real ``voxcpm`` package
when it is installed.
"""

import asyncio

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend import backends, models  # noqa: E402
from backend.backends import voxcpm2_backend as vb  # noqa: E402


def test_registered():
    cfg = next(c for c in backends.get_tts_model_configs() if c.engine == "voxcpm2")
    assert cfg.model_name == vb.VOXCPM2_MODEL_NAME == "voxcpm2"
    assert cfg.hf_repo_id == vb.VOXCPM2_HF_REPO == "openbmb/VoxCPM2"
    assert cfg.supports_instruct
    assert "voxcpm2" in backends.TTS_ENGINES
    assert isinstance(backends.get_tts_backend_for_engine("voxcpm2"), vb.VoxCPM2Backend)


def test_request_models_accept_engine():
    req = models.GenerationRequest(profile_id="p", text="hi", language="en", engine="voxcpm2")
    assert req.engine == "voxcpm2"


def test_xpu_patch_makes_resolver_accept_xpu():
    pytest.importorskip("voxcpm")
    import voxcpm.model.utils as u
    import voxcpm.model.voxcpm2 as m

    vb._xpu_patched = False
    vb._allow_xpu()
    assert u.resolve_runtime_device("xpu") == "xpu"
    assert m.resolve_runtime_device("xpu") == "xpu"
    assert m.resolve_runtime_device("cpu") == "cpu"


class FakeTTS:
    sample_rate = 48000


class FakeModel:
    def __init__(self):
        self.calls = []
        self.tts_model = FakeTTS()

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return np.zeros(480, dtype=np.float32)


def _backend():
    b = vb.VoxCPM2Backend()
    b._device = "cpu"
    b.model = FakeModel()
    return b


def test_generate_without_reference_uses_style_prefix():
    b = _backend()
    audio, sr = asyncio.run(b.generate("Hello", {}, instruct="calm (and) slow"))
    call = b.model.calls[0]
    assert sr == 48000 and audio.dtype == np.float32
    assert call["text"] == "(calm  and  slow)Hello"
    assert "reference_wav_path" not in call and "prompt_wav_path" not in call


def test_generate_with_reference_and_text_uses_ultimate_cloning():
    b = _backend()
    seen = {}

    orig = b.model.generate

    def spy(**kw):
        import os

        assert os.path.exists(kw["reference_wav_path"])
        seen["path"] = kw["reference_wav_path"]
        return orig(**kw)

    b.model.generate = spy
    prompt = {"ref_audio": torch.zeros(16000), "sample_rate": 16000, "ref_text": " hi there "}
    asyncio.run(b.generate("Hello", prompt))

    call = b.model.calls[0]
    assert call["prompt_text"] == "hi there"
    assert call["prompt_wav_path"] == call["reference_wav_path"]
    import os

    assert not os.path.exists(seen["path"]), "temp reference wav must be removed"


def test_generate_with_reference_only_is_controllable_cloning():
    b = _backend()
    prompt = {"ref_audio": torch.zeros(16000), "sample_rate": 16000, "ref_text": ""}
    asyncio.run(b.generate("Hello", prompt))
    call = b.model.calls[0]
    assert "reference_wav_path" in call and "prompt_wav_path" not in call
