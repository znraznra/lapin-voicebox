"""
Tests for xpu_ready(), the single XPU check behind device selection, /health
and the startup GPU status.

torch and intel_extension_for_pytorch are stubbed through sys.modules, so these
run without either installed and without any GPU.

Usage:
    python -m pytest backend/tests/test_xpu_ready.py -v
"""

import importlib.abc
import importlib.machinery
import sys
import types

import pytest

from backend.backends import base


class _Xpu:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _fake_torch(xpu_available=None, cuda_available=False):
    """xpu_available=None means torch has no `xpu` attribute at all (torch < 2.4)."""
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    if xpu_available is not None:
        t.xpu = _Xpu(xpu_available)
    return t


class _IpexFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves a fake intel_extension_for_pytorch whose import runs `on_import`."""

    def __init__(self, on_import):
        self._on_import = on_import

    def find_spec(self, name, path=None, target=None):
        if name == "intel_extension_for_pytorch":
            return importlib.machinery.ModuleSpec(name, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self._on_import()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(base.FORCE_CPU_ENV_VAR, raising=False)
    monkeypatch.delitem(sys.modules, "intel_extension_for_pytorch", raising=False)
    base.xpu_ready.cache_clear()
    yield
    base.xpu_ready.cache_clear()


@pytest.fixture
def usable(monkeypatch):
    monkeypatch.setattr(base, "xpu_usable", lambda: True)


def test_native_xpu_without_ipex_is_selected(monkeypatch, usable):
    """PyTorch 2.4+ XPU builds must work with no IPEX installed."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(xpu_available=True))
    monkeypatch.setitem(sys.modules, "intel_extension_for_pytorch", None)  # import raises

    assert base.xpu_ready() is True
    assert base.get_torch_device(allow_xpu=True) == "xpu"


def test_xpu_not_selected_unless_allowed(monkeypatch, usable):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(xpu_available=True))

    assert base.get_torch_device() == "cpu"


def test_visible_but_failing_xpu_falls_back_to_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(xpu_available=True))
    monkeypatch.setattr(base, "xpu_usable", lambda: False)

    assert base.xpu_ready() is False
    assert base.get_torch_device(allow_xpu=True) == "cpu"


def test_no_xpu_attribute_and_no_ipex(monkeypatch, usable):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(xpu_available=None))
    monkeypatch.setitem(sys.modules, "intel_extension_for_pytorch", None)

    assert base.xpu_ready() is False


def test_old_torch_gets_xpu_registered_by_ipex_import(monkeypatch, usable):
    """Pre-2.4 torch has no torch.xpu until IPEX is imported."""
    torch_stub = _fake_torch(xpu_available=None)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(
        sys,
        "meta_path",
        [_IpexFinder(lambda: setattr(torch_stub, "xpu", _Xpu(True)))] + sys.meta_path,
    )

    assert base.xpu_ready() is True


def test_broken_ipex_import_is_not_fatal(monkeypatch, usable):
    """A version-mismatched IPEX raising RuntimeError must not crash detection."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(xpu_available=True))

    def _boom():
        raise RuntimeError("IPEX/torch version mismatch")

    monkeypatch.setattr(sys, "meta_path", [_IpexFinder(_boom)] + sys.meta_path)

    assert base.xpu_ready() is True


def test_cuda_still_wins_over_xpu(monkeypatch, usable):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(xpu_available=True, cuda_available=True))

    assert base.get_torch_device(allow_xpu=True) == "cuda"
