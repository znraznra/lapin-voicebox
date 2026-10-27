"""
Tests for torch_load_default_cpu(), the workaround for chatterbox passing
map_location=None on non-CPU/MPS devices, which makes CUDA-saved checkpoints
fail to load on XPU/DirectML machines.

torch is stubbed through sys.modules, so these run without torch or a GPU.

Usage:
    python -m pytest backend/tests/test_torch_load_default_cpu.py -v
"""

import sys
import types

import pytest

from backend.backends.base import torch_load_default_cpu


@pytest.fixture
def fake_torch(monkeypatch):
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))

    t = types.ModuleType("torch")
    t.load = load
    monkeypatch.setitem(sys.modules, "torch", t)
    return t, load, calls


def test_explicit_none_map_location_becomes_cpu(fake_torch):
    """chatterbox passes map_location=None explicitly, so setdefault would not work."""
    t, _orig, calls = fake_torch

    with torch_load_default_cpu():
        t.load("s3gen.pt", map_location=None, weights_only=True)

    assert calls[0][1]["map_location"] == "cpu"
    assert calls[0][1]["weights_only"] is True


def test_missing_map_location_becomes_cpu(fake_torch):
    t, _orig, calls = fake_torch

    with torch_load_default_cpu():
        t.load("ve.pt")

    assert calls[0][1]["map_location"] == "cpu"


def test_explicit_map_location_is_kept(fake_torch):
    t, _orig, calls = fake_torch

    with torch_load_default_cpu():
        t.load("ve.pt", map_location="cuda:0")

    assert calls[0][1]["map_location"] == "cuda:0"


def test_positional_map_location_is_kept(fake_torch):
    t, _orig, calls = fake_torch

    with torch_load_default_cpu():
        t.load("ve.pt", "cuda:0")

    assert calls[0][0] == ("ve.pt", "cuda:0")
    assert "map_location" not in calls[0][1]


def test_torch_load_is_restored_even_on_error(fake_torch):
    t, orig, _calls = fake_torch

    with pytest.raises(RuntimeError):
        with torch_load_default_cpu():
            assert t.load is not orig
            raise RuntimeError("boom")

    assert t.load is orig
