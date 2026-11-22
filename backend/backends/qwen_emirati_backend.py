"""
Qwen3.5-TTS-Emirati backend.

Single-speaker Emirati Arabic fine-tune of Qwen3-TTS 1.7B Base
(HF: vadimbelsky/qwen3.5-TTS-Emirati). It loads through the same
``qwen_tts`` package as the other Qwen engines and is driven through
``generate_custom_voice`` with the fixed speaker ``emirati_speaker``.
No reference audio is needed.

Written from the model card only: neither config.json nor infer.py was
readable when this was added, so the speaker id, language name and 24 kHz
output follow the card and have not been run against the weights.
"""

import asyncio
import logging
from typing import Optional

import numpy as np
import torch

from .base import (
    is_model_cached,
    get_torch_device,
    empty_device_cache,
    manual_seed,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

QWEN_EMIRATI_HF_REPO = "vadimbelsky/qwen3.5-TTS-Emirati"
QWEN_EMIRATI_MODEL_NAME = "qwen-emirati"
QWEN_EMIRATI_SPEAKER = "emirati_speaker"

# (speaker_id, display_name, gender, native_language_code, description)
# The model card does not state a gender, so none is asserted.
QWEN_EMIRATI_VOICES = [
    (
        QWEN_EMIRATI_SPEAKER,
        "Emirati speaker",
        "unspecified",
        "ar",
        "Emirati Arabic single-speaker fine-tune of Qwen3-TTS 1.7B",
    ),
]

# Local on purpose: the shared LANGUAGE_CODE_TO_NAME is used by other engines.
_LANGUAGE_NAMES = {"ar": "Arabic"}


class QwenEmiratiBackend:
    """Qwen3.5-TTS-Emirati backend — one fixed Emirati Arabic speaker."""

    def __init__(self):
        self.model = None
        self.model_size = "default"
        self.device = self._get_device()
        self._lock = asyncio.Lock()

    def _get_device(self) -> str:
        return get_torch_device(allow_xpu=True, allow_directml=True)

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: Optional[str] = None) -> str:
        return QWEN_EMIRATI_HF_REPO

    def _is_model_cached(self, model_size: Optional[str] = None) -> bool:
        return is_model_cached(QWEN_EMIRATI_HF_REPO)

    async def load_model(self, model_size: str = "default") -> None:
        async with self._lock:
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_model_sync)

    load_model_async = load_model

    def _load_model_sync(self) -> None:
        is_cached = self._is_model_cached()

        with model_load_progress(QWEN_EMIRATI_MODEL_NAME, is_cached):
            from qwen_tts import Qwen3TTSModel

            logger.info("Loading Qwen Emirati on %s...", self.device)

            if self.device == "cpu":
                self.model = Qwen3TTSModel.from_pretrained(
                    QWEN_EMIRATI_HF_REPO,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                )
            else:
                self.model = Qwen3TTSModel.from_pretrained(
                    QWEN_EMIRATI_HF_REPO,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16,
                )

        logger.info("Qwen Emirati loaded successfully")

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            empty_device_cache(self.device)
            logger.info("Qwen Emirati unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """The model has one built-in speaker; reference audio is ignored."""
        return {
            "voice_type": "preset",
            "preset_engine": "qwen_emirati",
            "preset_voice_id": QWEN_EMIRATI_SPEAKER,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "ar",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """Generate Emirati Arabic speech. ``instruct`` is not used."""
        await self.load_model()

        def _generate_sync():
            if seed is not None:
                manual_seed(seed, self.device)

            with torch.inference_mode():
                wavs, sample_rate = self.model.generate_custom_voice(
                    text=text,
                    language=_LANGUAGE_NAMES.get(language, "Auto"),
                    speaker=QWEN_EMIRATI_SPEAKER,
                )
            return wavs[0], sample_rate

        return await asyncio.to_thread(_generate_sync)
