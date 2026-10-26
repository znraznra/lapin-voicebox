"""Intel GPU detection used for setup hints and the /health warning."""

from pathlib import Path
from unittest.mock import patch

from backend.utils import platform_detect as pd


def _fake_drm(root: Path, vendors: list[str]) -> Path:
    for i, vendor in enumerate(vendors):
        dev = root / f"card{i}" / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text(vendor + "\n")
    # Connector entries such as card0-DP-1 must not be mistaken for GPUs.
    (root / "card0-DP-1").mkdir()
    return root


def test_linux_finds_intel_gpu(tmp_path):
    drm = _fake_drm(tmp_path, ["0x8086"])
    with patch.object(pd.subprocess, "run", side_effect=FileNotFoundError):
        assert pd._linux_intel_gpu_names(drm) == ("Intel GPU",)


def test_linux_ignores_other_vendors(tmp_path):
    drm = _fake_drm(tmp_path, ["0x10de", "0x1002"])
    assert pd._linux_intel_gpu_names(drm) == ()


def test_linux_hybrid_laptop_still_reports_intel(tmp_path):
    drm = _fake_drm(tmp_path, ["0x8086", "0x10de"])
    with patch.object(pd.subprocess, "run", side_effect=FileNotFoundError):
        assert len(pd._linux_intel_gpu_names(drm)) == 1


def test_linux_uses_lspci_name(tmp_path):
    drm = _fake_drm(tmp_path, ["0x8086"])

    class Result:
        stdout = '00:02.0 "VGA compatible controller" "Intel Corporation" "Arc B580" -r08 "Intel" "x"\n'

    with patch.object(pd.subprocess, "run", return_value=Result()):
        assert pd._linux_intel_gpu_names(drm) == ("Arc B580",)


def test_missing_drm_dir_is_empty(tmp_path):
    assert pd._linux_intel_gpu_names(tmp_path / "nope") == ()


def test_warning_is_none_without_intel_gpu():
    pd.intel_gpu_names.cache_clear()
    with patch.object(pd, "intel_gpu_names", return_value=()):
        assert pd.intel_gpu_warning() is None


def test_warning_differs_per_os():
    with patch.object(pd, "intel_gpu_names", return_value=("Arc A770",)):
        with patch.object(pd.platform, "system", return_value="Linux"):
            linux = pd.intel_gpu_warning()
        with patch.object(pd.platform, "system", return_value="Windows"):
            windows = pd.intel_gpu_warning()
    assert pd.XPU_INDEX_URL in linux and "Level Zero" in linux
    assert pd.XPU_INDEX_URL in windows and "requirements-directml.txt" in windows
