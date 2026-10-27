"""
Tests for unloading other TTS engines when switching engines.

Without this, every engine used stays resident, and switching between large
models on a 12 GB card runs out of VRAM ("XPU out of memory" while loading
TADA with another engine still loaded).

Backends are stubbed, so these run without torch or a GPU.

Usage:
    python -m pytest backend/tests/test_engine_switch_eviction.py -v
"""

import sys
import types

import pytest

from backend import backends


class FakeBackend:
    def __init__(self, loaded=False, fail_load=False, fail_unload=False):
        self.loaded = loaded
        self.fail_load = fail_load
        self.fail_unload = fail_unload
        self.load_calls = 0
        self.unload_calls = 0

    def is_loaded(self):
        return self.loaded

    async def load_model(self, *args, **kwargs):
        self.load_calls += 1
        if self.fail_load:
            raise RuntimeError("load failed")
        self.loaded = True

    def unload_model(self):
        self.unload_calls += 1
        if self.fail_unload:
            raise RuntimeError("unload failed")
        self.loaded = False


@pytest.fixture
def engines(monkeypatch):
    registry = {
        "kokoro": FakeBackend(loaded=True),
        "omnivoice": FakeBackend(loaded=True),
        "luxtts": FakeBackend(loaded=False),
    }
    monkeypatch.setattr(backends, "_tts_backends", registry)
    monkeypatch.setattr(backends, "_engine_leases", {})
    monkeypatch.delenv(backends.KEEP_MODELS_LOADED_ENV_VAR, raising=False)

    cache = types.ModuleType("backend.utils.cache")
    cache.cleared = 0

    def clear():
        cache.cleared += 1

    cache.clear_voice_prompt_memory_cache = clear
    monkeypatch.setitem(sys.modules, "backend.utils.cache", cache)
    return registry, cache


@pytest.mark.asyncio
async def test_switching_unloads_other_loaded_engines(engines):
    registry, cache = engines

    await backends.load_engine_model("luxtts")

    assert registry["luxtts"].loaded
    assert not registry["kokoro"].loaded
    assert not registry["omnivoice"].loaded
    assert cache.cleared == 1


@pytest.mark.asyncio
async def test_engine_in_use_is_not_evicted(engines):
    registry, _cache = engines

    await backends.load_engine_model("kokoro")  # a request is now using kokoro
    await backends.load_engine_model("luxtts")  # a second request switches engines

    assert registry["kokoro"].loaded, "must not unload a model mid-generation"
    assert not registry["omnivoice"].loaded

    backends.release_engine("kokoro")
    registry["luxtts"].loaded = False
    await backends.load_engine_model("luxtts")

    assert not registry["kokoro"].loaded, "evictable again once released"


@pytest.mark.asyncio
async def test_already_loaded_target_leaves_others_alone(engines):
    registry, cache = engines

    await backends.load_engine_model("kokoro")

    assert registry["omnivoice"].loaded
    assert registry["kokoro"].load_calls == 1
    assert cache.cleared == 0


@pytest.mark.asyncio
async def test_opt_out_keeps_everything_loaded(engines, monkeypatch):
    registry, _cache = engines
    monkeypatch.setenv(backends.KEEP_MODELS_LOADED_ENV_VAR, "1")

    await backends.load_engine_model("luxtts")

    assert registry["kokoro"].loaded
    assert registry["omnivoice"].loaded
    assert registry["luxtts"].loaded


@pytest.mark.asyncio
async def test_failed_load_releases_its_lease(engines):
    registry, _cache = engines
    registry["luxtts"].fail_load = True

    with pytest.raises(RuntimeError):
        await backends.load_engine_model("luxtts")

    assert "luxtts" not in backends._engine_leases


@pytest.mark.asyncio
async def test_unload_failure_does_not_block_the_switch(engines):
    registry, _cache = engines
    registry["kokoro"].fail_unload = True

    await backends.load_engine_model("luxtts")

    assert registry["luxtts"].loaded
    assert not registry["omnivoice"].loaded


@pytest.mark.asyncio
async def test_leases_are_counted_per_request(engines):
    await backends.load_engine_model("kokoro")
    await backends.load_engine_model("kokoro")
    assert backends._engine_leases["kokoro"] == 2

    backends.release_engine("kokoro")
    assert backends._engine_leases["kokoro"] == 1

    backends.release_engine("kokoro")
    assert "kokoro" not in backends._engine_leases


def test_release_without_a_lease_is_harmless(engines):
    backends.release_engine("never-leased")

    assert backends._engine_leases == {}
