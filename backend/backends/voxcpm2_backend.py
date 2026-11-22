"""
VoxCPM2 backend implementation.

Wraps OpenBMB's VoxCPM2: a 2B-parameter tokenizer-free TTS (Apache-2.0
weights and code) with 30 languages, 48 kHz output, zero-shot cloning and
free-form style control. The language is detected from the text, so the
``language`` argument is not forwarded.

Voice handling:
  - Reference audio only      -> "controllable cloning" (reference_wav_path)
  - Reference audio + text    -> "ultimate cloning" (also prompt_wav_path +
                                 prompt_text), the most faithful mode
  - ``instruct``              -> upstream's parenthesised style prefix,
                                 e.g. "(slightly faster, cheerful)Hello"

Device notes:
  - Upstream's device resolver only knows cuda / mps / cpu. For Intel Arc we
    patch it to accept "xpu" (see ``_allow_xpu``); that path is not covered by
    upstream and has not been run on hardware here.
  - ``optimize=False`` always: upstream's torch.compile path is CUDA/triton
    only and adds a long first-run warm-up.
  - Installed with ``--no-deps`` (the package pins gradio, funasr, modelscope,
    datasets<4, ...). Inference only needs the packages in requirements.txt.
"""

import asyncio
import logging
import os
import tempfile
import threading
from typing import Optional, Tuple

import numpy as np

from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    empty_device_cache,
    get_torch_device,
    is_model_cached,
    manual_seed,
    model_load_progress,
)
from ..utils.cache import cache_voice_prompt, get_cache_key, get_cached_voice_prompt

logger = logging.getLogger(__name__)

VOXCPM2_HF_REPO = "openbmb/VoxCPM2"
VOXCPM2_MODEL_NAME = "voxcpm2"

# Upstream wants 16 kHz reference audio.
REFERENCE_SAMPLE_RATE = 16000
DEFAULT_SAMPLE_RATE = 48000

# Upstream's own recommended defaults.
CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 10

_PROMPT_FORMAT_VERSION = 1

_xpu_patch_lock = threading.Lock()
_xpu_patched = False


def _allow_xpu() -> None:
    """Teach VoxCPM's device resolver about Intel XPU (idempotent)."""
    global _xpu_patched
    with _xpu_patch_lock:
        if _xpu_patched:
            return

        import importlib

        utils = importlib.import_module("voxcpm.model.utils")
        original = utils.resolve_runtime_device

        def resolve(device, configured_device="cuda"):
            if device is not None and str(device).strip().lower().startswith("xpu"):
                return str(device).strip().lower()
            return original(device, configured_device)

        # voxcpm2.py does `from .utils import resolve_runtime_device`, so the
        # name has to be replaced in every module that imported it.
        for mod_name in ("voxcpm.model.utils", "voxcpm.model.voxcpm2", "voxcpm.model.voxcpm"):
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "resolve_runtime_device"):
                mod.resolve_runtime_device = resolve
        _xpu_patched = True


class VoxCPM2Backend:
    """VoxCPM2 backend for multilingual cloning and style-controlled TTS."""

    def __init__(self):
        self.model = None
        self.model_size = "default"  # single checkpoint
        self._device = None
        self._model_load_lock = asyncio.Lock()

    def _get_device(self) -> str:
        # On MPS upstream switches to fp32 by itself (low precision is glitchy
        # there). DirectML is not allowed: upstream can't use it.
        return get_torch_device(allow_xpu=True, allow_mps=True)

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._get_device()
        return self._device

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return VOXCPM2_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(VOXCPM2_HF_REPO)

    async def load_model(self, model_size: str = "default") -> None:
        if self.model is not None:
            return
        async with self._model_load_lock:
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    def _load_model_sync(self) -> None:
        is_cached = self._is_model_cached()

        with model_load_progress(VOXCPM2_MODEL_NAME, is_cached):
            try:
                from voxcpm import VoxCPM
            except ImportError as exc:
                raise RuntimeError(
                    "VoxCPM2 needs the `voxcpm` package. Install it with "
                    "`pip install --no-deps voxcpm einops` (see backend/requirements.txt)."
                ) from exc

            device = self.device
            if device == "xpu":
                _allow_xpu()
            elif device not in ("cuda", "mps", "cpu"):
                # e.g. DirectML: not supported by upstream.
                device = "cpu"

            logger.info("Loading VoxCPM2 on %s...", device)
            self.model = VoxCPM.from_pretrained(
                VOXCPM2_HF_REPO,
                load_denoiser=False,
                optimize=False,
                device=device,
            )
            self._device = device

        logger.info("VoxCPM2 loaded successfully")

    def unload_model(self) -> None:
        if self.model is not None:
            device = self.device
            del self.model
            self.model = None
            self._device = None
            import gc

            gc.collect()
            empty_device_cache(device)
            logger.info("VoxCPM2 unloaded")

    @property
    def sample_rate(self) -> int:
        try:
            return int(self.model.tts_model.sample_rate)
        except Exception:
            return DEFAULT_SAMPLE_RATE

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """Load the reference clip as 16 kHz mono and keep it with its transcript.

        VoxCPM builds its own prompt cache inside generate(), so the "prompt"
        is the waveform itself. It is stored as a tensor (not a path) because
        profile audio can be a temporary combined file, and tensors survive
        the on-disk cache's ``weights_only`` load.
        """
        cache_key = (
            ("voxcpm2_" + get_cache_key(audio_path, reference_text)) if use_cache else None
        )
        if cache_key:
            cached = get_cached_voice_prompt(cache_key)
            if (
                isinstance(cached, dict)
                and cached.get("format_version") == _PROMPT_FORMAT_VERSION
                and "ref_audio" in cached
            ):
                return cached, True

        def _load_sync() -> dict:
            import torch

            from ..utils.audio import load_audio

            audio, _sr = load_audio(str(audio_path), sample_rate=REFERENCE_SAMPLE_RATE)
            return {
                "format_version": _PROMPT_FORMAT_VERSION,
                "ref_audio": torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)),
                "sample_rate": REFERENCE_SAMPLE_RATE,
                "ref_text": (reference_text or "").strip(),
            }

        prompt = await asyncio.to_thread(_load_sync)

        if cache_key:
            cache_voice_prompt(cache_key, prompt)

        return prompt, False

    async def combine_voice_prompts(self, audio_paths, reference_texts):
        return await _combine_voice_prompts(
            audio_paths, reference_texts, sample_rate=REFERENCE_SAMPLE_RATE
        )

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio with VoxCPM2.

        Args:
            text: Text to synthesize (any supported language, auto-detected).
            voice_prompt: Dict from create_voice_prompt(). Without a
                reference clip the model picks its own voice.
            language: Unused; VoxCPM2 needs no language tag.
            seed: Random seed for reproducibility.
            instruct: Free-form style description, sent as a "(...)" prefix.
        """
        await self.load_model()

        ref_audio = (voice_prompt or {}).get("ref_audio")
        ref_text = ((voice_prompt or {}).get("ref_text") or "").strip()
        ref_sr = int((voice_prompt or {}).get("sample_rate") or REFERENCE_SAMPLE_RATE)

        # Parentheses delimit the style prefix upstream; strip them from the
        # user's text so the instruction can't be broken out of or injected.
        spoken = text
        if instruct and instruct.strip():
            style = instruct.strip().replace("(", " ").replace(")", " ")
            spoken = f"({style}){text}"

        def _generate_sync() -> Tuple[np.ndarray, int]:
            import soundfile as sf

            if seed is not None:
                manual_seed(seed, self.device)

            tmp_path = None
            try:
                kwargs = {
                    "text": spoken,
                    "cfg_value": CFG_VALUE,
                    "inference_timesteps": INFERENCE_TIMESTEPS,
                }
                if ref_audio is not None:
                    wav = ref_audio.numpy() if hasattr(ref_audio, "numpy") else np.asarray(ref_audio)
                    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    sf.write(tmp_path, wav, ref_sr)
                    kwargs["reference_wav_path"] = tmp_path
                    if ref_text:
                        kwargs["prompt_wav_path"] = tmp_path
                        kwargs["prompt_text"] = ref_text

                audio = self.model.generate(**kwargs)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

            return np.asarray(audio, dtype=np.float32), self.sample_rate

        return await asyncio.to_thread(_generate_sync)
