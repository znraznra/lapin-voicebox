"""
Platform detection for backend selection.
"""

import os
import platform
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Literal


def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon (arm64 macOS).

    Returns:
        True if on Apple Silicon, False otherwise
    """
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@lru_cache(maxsize=1)
def is_amd_gpu_windows() -> bool:
    """
    Check if the primary GPU on Windows is an AMD Radeon card.

    Uses WMI to query Win32_VideoController, with a fallback to
    torch.cuda.get_device_name(0) if WMI is unavailable.  This is
    useful for deciding whether the ROCm backend is appropriate.

    Result is cached since it shells out to PowerShell and the GPU
    does not change at runtime — safe to call from the health path.

    Returns:
        True if an AMD GPU is detected on Windows, False otherwise.
    """
    if platform.system() != "Windows":
        return False

    # Primary method: WMI query for AMD adapters
    try:
        result = subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "Where-Object {$_.AdapterCompatibility -like '*AMD*'} | "
                "Measure-Object | Select-Object -ExpandProperty Count",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if int(result.stdout.strip()) > 0:
            return True
    except Exception:
        pass

    # Fallback: torch.cuda.get_device_name(0) (works for ROCm/HIP too)
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            if "Radeon" in name or "AMD" in name:
                return True
    except Exception:
        pass

    return False


_INTEL_PCI_VENDOR = "0x8086"
XPU_INDEX_URL = "https://download.pytorch.org/whl/xpu"


def _linux_intel_gpu_names(drm_root: Path = Path("/sys/class/drm")) -> tuple[str, ...]:
    """Intel GPUs on Linux, found via the kernel's DRM sysfs entries.

    Works without any GPU userspace installed, so it can run before XPU torch
    is present. Names come from lspci when available, else a generic label.
    """
    found = 0
    try:
        for vendor_file in sorted(drm_root.glob("card[0-9]*/device/vendor")):
            if vendor_file.read_text().strip().lower() == _INTEL_PCI_VENDOR:
                found += 1
    except OSError:
        return ()
    if not found:
        return ()

    names: list[str] = []
    try:
        out = subprocess.run(["lspci", "-mm"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if re.search(r'"(VGA compatible controller|Display controller|3D controller)"', line) and "Intel" in line:
                parts = re.findall(r'"([^"]*)"', line)
                if len(parts) >= 3:
                    names.append(parts[2])
    except Exception:
        pass
    return tuple(names) if names else tuple("Intel GPU" for _ in range(found))


def _windows_intel_gpu_names() -> tuple[str, ...]:
    """Intel GPUs on Windows, via the same WMI query the setup recipe uses."""
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
    except Exception:
        return ()
    return tuple(n.strip() for n in out.splitlines() if "Intel" in n)


@lru_cache(maxsize=1)
def intel_gpu_names() -> tuple[str, ...]:
    """Names of Intel GPUs on this host (empty tuple if none, or on macOS).

    Cached: it shells out and the hardware does not change at runtime.
    """
    system = platform.system()
    if system == "Linux":
        return _linux_intel_gpu_names()
    if system == "Windows":
        return _windows_intel_gpu_names()
    return ()


# PyTorch's XPU build officially targets Arc (A/B series), Core Ultra "Arc
# Graphics" iGPUs (Meteor Lake and later) and Data Center GPU Max. Iris Xe / UHD
# (Gen12 and older) are not listed, so they should not be sent down the XPU path.
_XPU_SUPPORTED = re.compile(r"\barc\b|battlemage|alchemist|data center gpu max|ponte vecchio", re.I)


def is_xpu_supported_gpu(name: str) -> bool:
    """Whether PyTorch's XPU build officially supports this Intel GPU name."""
    return bool(_XPU_SUPPORTED.search(name))


def intel_gpu_warning() -> str | None:
    """Explain why an installed Intel GPU is not being used, per OS.

    Call only when no GPU backend (CUDA/XPU/DirectML) is active. Returns None
    when there is no Intel GPU to talk about.
    """
    names = intel_gpu_names()
    if not names:
        return None
    gpu = names[0]
    install = f"pip install torch torchaudio --index-url {XPU_INDEX_URL}"
    on_linux = platform.system() == "Linux"

    if not any(is_xpu_supported_gpu(n) for n in names):
        # Iris Xe / UHD: installing the XPU build would not help.
        msg = (
            f"Intel GPU detected ({gpu}) but PyTorch's XPU build does not officially "
            "support Iris Xe / UHD integrated graphics, so generation runs on CPU."
        )
        if not on_linux:
            msg += " An experimental DirectML option is described in backend/requirements-directml.txt."
        return msg

    if on_linux:
        return (
            f"Intel GPU detected ({gpu}) but this PyTorch build cannot use it, "
            f"so generation runs on CPU. Install the XPU build: {install}. "
            "You also need the Intel compute runtime and Level Zero loader from your "
            "distro, and your user in the 'render' group."
        )
    return (
        f"Intel GPU detected ({gpu}) but this PyTorch build cannot use it, "
        f"so generation runs on CPU. Install the XPU build: {install}."
    )


# Platform tokens for which we actually build and publish downloadable GPU
# server assets. Anything else must not be offered a download (it would 404).
SUPPORTED_GPU_ASSET_PLATFORMS = ("linux-x86_64", "windows-x86_64")


@lru_cache(maxsize=1)
def is_amd_rocm_capable() -> bool:
    """Whether this host can run the downloadable ROCm backend.

    Decides whether to offer the ROCm backend download in the UI, so it must
    work before ROCm torch is installed (can't rely on torch.version.hip).

    Windows: reuse the WMI/torch AMD probe. Linux: /dev/kfd is created by the
    amdgpu kernel driver whenever ROCm compute is available — the same gate the
    Docker entrypoint and setup recipe use. Gated on a platform we publish
    assets for so unsupported arches (e.g. linux-arm64) aren't offered a 404.
    """
    if server_asset_platform() not in SUPPORTED_GPU_ASSET_PLATFORMS:
        return False
    system = platform.system()
    if system == "Windows":
        return is_amd_gpu_windows()
    if system == "Linux":
        return os.path.exists("/dev/kfd")
    return False


def server_asset_platform() -> str:
    """Platform token identifying which downloadable GPU server assets to fetch.

    Windows and Linux ROCm/CUDA builds are distinct binaries published as
    separate release assets, so the download URL must be qualified by platform.
    Only the platforms that ship a downloadable GPU backend are meaningful here
    (macOS uses MLX/Metal and has no CUDA/ROCm server to download).
    """
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64"
    return f"{platform.system().lower()}-{arch}"


def get_backend_type() -> Literal["mlx", "pytorch"]:
    """
    Detect the best backend for the current platform.

    Returns:
        "mlx" on Apple Silicon (if MLX is available and functional), "pytorch" otherwise
    """
    if is_apple_silicon():
        try:
            import mlx.core  # noqa: F401 — triggers native lib loading
            return "mlx"
        except (ImportError, OSError, RuntimeError):
            # MLX not installed, or native libraries failed to load inside a
            # PyInstaller bundle (OSError on missing .dylib / .metallib).
            # Fall through to PyTorch.
            return "pytorch"
    return "pytorch"
