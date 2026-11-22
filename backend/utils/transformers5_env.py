"""
Private transformers 5 install for engines that need it (Qwen3.5).

Voicebox pins ``transformers<=4.57.6`` because other engines depend on 4.57.x
internals, but Qwen3.5 only exists in transformers 5. Rather than move the pin,
a second copy of transformers 5 (plus the newer huggingface_hub / tokenizers it
requires) lives in its own directory and is used only by a worker subprocess
(``backends/qwen35_worker.py``). The main process never imports it.

Locations searched, in order: ``$VOICEBOX_TF5_DIR``, ``backend/.transformers5``
(created by ``just setup``), ``<data dir>/transformers5`` (created on first
use). Run ``python -m backend.utils.transformers5_env`` to install it.
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

REQUIREMENT = "transformers>=5"
MIN_PYTHON = (3, 11)  # transformers 5 requires Python >= 3.11
ENV_VAR = "VOICEBOX_TF5_DIR"
_MARKER = "transformers"


def repo_dir() -> Path:
    return Path(__file__).resolve().parent.parent / ".transformers5"


def data_dir() -> Path:
    from .. import config

    return config.get_data_dir() / "transformers5"


def _is_valid(path: Path) -> bool:
    return (path / _MARKER / "models" / "qwen3_5").is_dir()


def find_installed() -> Optional[Path]:
    import os

    candidates = []
    if os.environ.get(ENV_VAR):
        candidates.append(Path(os.environ[ENV_VAR]))
    candidates.append(repo_dir())
    try:
        candidates.append(data_dir())
    except Exception:
        pass
    for path in candidates:
        if _is_valid(path):
            return path
    return None


def can_install() -> Optional[str]:
    """Return a reason string if installing here is impossible, else None."""
    if getattr(sys, "frozen", False):
        return (
            "Qwen3.5 is not available in the packaged app (it needs a Python "
            "environment to run its own transformers 5 install). Run Voicebox "
            "from source (`just setup`, `just dev`) to use it."
        )
    if sys.version_info < MIN_PYTHON:
        return (
            f"Qwen3.5 needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ (transformers 5), "
            f"but this is Python {sys.version_info[0]}.{sys.version_info[1]}."
        )
    return None


def install(target: Path) -> Path:
    """pip-install transformers 5 into ``target`` and prune what the main env supplies."""
    reason = can_install()
    if reason:
        raise RuntimeError(reason)

    target = Path(target)
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Installing transformers 5 into %s ...", target)
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--target", str(target), REQUIREMENT],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError(
            "Installing transformers 5 for Qwen3.5 failed:\n" + (proc.stderr or proc.stdout)[-1500:]
        )

    # numpy is compiled code shared with torch; the worker must use the main
    # environment's copy (transformers only needs numpy>=1.17).
    for child in list(target.iterdir()):
        if child.name.lower().startswith("numpy") or child.name == "bin":
            shutil.rmtree(child, ignore_errors=True)

    if not _is_valid(target):
        raise RuntimeError("transformers 5 installed but has no qwen3_5 model support")
    return target


def ensure_installed() -> Path:
    """Return the transformers 5 directory, installing it first if needed."""
    found = find_installed()
    if found:
        return found
    return install(data_dir())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reason = can_install()
    if reason:
        # Not fatal for setup: every other engine works without it.
        print(f"Skipping Qwen3.5 support: {reason}")
        sys.exit(0)
    try:
        print(f"Installed: {install(repo_dir())}")
    except RuntimeError as exc:
        print(f"Skipping Qwen3.5 support: {exc}")
