"""
Qwen3 LLM backend implementations.

Provides MLX (Apple Silicon, 4-bit community quants) and PyTorch
(transformers AutoModelForCausalLM) paths that share the same
`LLMBackend` protocol and model-load progress plumbing as the TTS
and STT engines.
"""

import asyncio
import logging
from typing import Optional

from . import LLMBackend, DEFAULT_LLM_MAX_TOKENS, DEFAULT_LLM_TEMPERATURE, llm_progress_name as progress_name
from .base import (
    is_model_cached,
    get_torch_device,
    empty_device_cache,
    empty_mlx_cache,
    manual_seed,
    model_load_progress,
)

logger = logging.getLogger(__name__)


PYTORCH_HF_REPOS = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B",
    "4B": "Qwen/Qwen3-4B",
    # Qwen3.5 needs transformers>=5 to load (see _require_qwen3_5_support).
    # No MLX repo is registered for it.
    "3.5-0.8B": "Qwen/Qwen3.5-0.8B",
    "3.5-2B": "Qwen/Qwen3.5-2B",
    "3.5-4B": "Qwen/Qwen3.5-4B",
}

MLX_HF_REPOS = {
    "0.6B": "mlx-community/Qwen3-0.6B-4bit",
    "1.7B": "mlx-community/Qwen3-1.7B-4bit",
    "4B": "mlx-community/Qwen3-4B-4bit",
}


def _is_qwen3_5(model_size: str) -> bool:
    return model_size.startswith("3.5-")


def _require_qwen3_5_support() -> None:
    """Fail clearly when the installed transformers predates Qwen3.5."""
    from importlib.util import find_spec

    try:
        supported = find_spec("transformers.models.qwen3_5") is not None
    except (ImportError, ValueError):
        supported = False
    if not supported:
        import transformers

        raise RuntimeError(
            "Qwen3.5 needs transformers>=5 to load, but transformers "
            f"{transformers.__version__} is installed. The weights are downloaded; "
            "upgrade with `pip install -U 'transformers>=5'` (other engines are "
            "pinned to 4.57.x and may not work after that)."
        )


def _build_messages(
    prompt: str,
    system: Optional[str],
    examples: Optional[list[tuple[str, str]]] = None,
) -> list[dict]:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if examples:
        for user_text, assistant_text in examples:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": prompt})
    return messages


class PyTorchQwenLLMBackend:
    """Qwen3 LLM backend using HuggingFace transformers."""

    def __init__(self, model_size: str = "0.6B"):
        self.model = None
        self.tokenizer = None
        self.model_size = model_size
        self._current_model_size: Optional[str] = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        return get_torch_device(allow_xpu=True, allow_directml=True, allow_mps=True)

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        if model_size not in PYTORCH_HF_REPOS:
            raise ValueError(f"Unknown Qwen3 size: {model_size}")
        return PYTORCH_HF_REPOS[model_size]

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(self._get_model_path(model_size))

    async def load_model(self, model_size: Optional[str] = None) -> None:
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self._current_model_size == model_size:
            return

        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()

        await asyncio.to_thread(self._load_model_sync, model_size)

    def _load_model_sync(self, model_size: str) -> None:
        import torch

        progress_model_name = progress_name(model_size)
        is_cached = self._is_model_cached(model_size)
        repo = self._get_model_path(model_size)

        with model_load_progress(progress_model_name, is_cached):
            if _is_qwen3_5(model_size):
                # Download first so the weights are on disk even if this
                # transformers version can't load the architecture.
                from huggingface_hub import snapshot_download

                snapshot_download(repo)
                _require_qwen3_5_support()
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("Loading Qwen3 %s on %s...", model_size, self.device)
            # Loads run with the process's default HF_HUB_OFFLINE state.
            # Forcing offline for cached models flips process-global state
            # and silently switches every concurrent download/load on other
            # threads to offline mode (issue #841) — the same regression
            # removed app-wide in #524/#530.
            self.tokenizer = AutoTokenizer.from_pretrained(repo)
            if self.device in ("cuda", "xpu", "mps"):
                # Qwen3.5 is trained in bf16; fp16 risks overflow.
                dtype = torch.bfloat16 if _is_qwen3_5(model_size) and self.device != "mps" else torch.float16
            else:
                dtype = torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                repo,
                dtype=dtype,
            )
            self.model.to(self.device)
            self.model.eval()

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("Qwen3 %s loaded successfully", model_size)

    def unload_model(self) -> None:
        if self.model is None:
            return
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._current_model_size = None
        empty_device_cache(self.device)
        logger.info("Qwen3 unloaded")

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        model_size: Optional[str] = None,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        await self.load_model(model_size)
        return await asyncio.to_thread(
            self._generate_sync, prompt, system, max_tokens, temperature, examples
        )

    def _generate_sync(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        import torch

        messages = _build_messages(prompt, system, examples)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        do_sample = temperature > 0
        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = 0.9

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0, input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class MLXQwenLLMBackend:
    """Qwen3 LLM backend using mlx-lm (Apple Silicon)."""

    def __init__(self, model_size: str = "0.6B"):
        self.model = None
        self.tokenizer = None
        self.model_size = model_size
        self._current_model_size: Optional[str] = None
        # Same role as MLXTTSBackend._op_lock: keeps two coroutines from
        # racing their reload decisions around one load-then-generate.
        self._op_lock = asyncio.Lock()

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        if model_size not in MLX_HF_REPOS:
            raise ValueError(f"Unknown Qwen3 size: {model_size}")
        return MLX_HF_REPOS[model_size]

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(
            self._get_model_path(model_size),
            weight_extensions=(".safetensors", ".bin", ".npz"),
        )

    async def load_model(self, model_size: Optional[str] = None) -> None:
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self._current_model_size == model_size:
            return

        # Routed through the same dedicated MLX thread as TTS/STT — MLX's
        # Metal stream is thread-local, so load and generate must run on
        # one thread (see mlx_backend._run_on_mlx_thread / issue #699).
        from .mlx_backend import _run_on_mlx_thread

        async with self._op_lock:
            await _run_on_mlx_thread(self._reload_sync, model_size)

    def _reload_sync(self, model_size: str) -> None:
        """Unload a mismatched model and load the requested one, in one MLX-thread op."""
        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()
        self._load_model_sync(model_size)

    def _load_model_sync(self, model_size: str) -> None:
        from mlx_lm import load as mlx_load

        progress_model_name = progress_name(model_size)
        is_cached = self._is_model_cached(model_size)
        repo = self._get_model_path(model_size)

        with model_load_progress(progress_model_name, is_cached):
            logger.info("Loading Qwen3 %s via MLX...", model_size)
            # See the PyTorch loader comment — no offline forcing (issue #841).
            loaded = mlx_load(repo)

        # mlx_lm.load returns (model, tokenizer) by default and
        # (model, tokenizer, config) when return_config=True.
        self.model = loaded[0]
        self.tokenizer = loaded[1]

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("Qwen3 %s (MLX) loaded successfully", model_size)

    def unload_model(self) -> None:
        if self.model is None:
            return
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._current_model_size = None
        empty_mlx_cache()
        logger.info("Qwen3 (MLX) unloaded")

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
        temperature: float = DEFAULT_LLM_TEMPERATURE,
        model_size: Optional[str] = None,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        from .mlx_backend import _run_on_mlx_thread

        resolved_size = model_size if model_size is not None else self.model_size

        def _reload_and_generate_sync() -> str:
            # One worker submission for load + generate, as in
            # MLXTTSBackend._reload_and_generate_sync: no gap between the two
            # where another request's load_model (different size) or an
            # unload can swap or null out self.model / self.tokenizer.
            if self.model is None or self._current_model_size != resolved_size:
                self._reload_sync(resolved_size)
            try:
                return self._generate_sync(prompt, system, max_tokens, temperature, examples)
            finally:
                # Drain the MLX pool if an unload landed mid-generation (see
                # MLXTTSBackend._reload_and_generate_sync).
                if self.model is None:
                    empty_mlx_cache()

        async with self._op_lock:
            return await _run_on_mlx_thread(_reload_and_generate_sync)

    def _generate_sync(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        examples: Optional[list[tuple[str, str]]] = None,
    ) -> str:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        model, tokenizer = self.model, self.tokenizer
        try:
            messages = _build_messages(prompt, system, examples)
            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            sampler = make_sampler(temp=temperature, top_p=0.9) if temperature > 0 else None
            text = mlx_generate(
                model,
                tokenizer,
                prompt=chat_prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )
            return text.strip()
        finally:
            # Drop the local binding here, inside the frame a propagating
            # traceback would keep alive, so the caller's drain can
            # actually return the model's buffers to MLX.
            del model, tokenizer
