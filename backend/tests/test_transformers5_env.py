"""transformers 5 side-install used by Qwen3.5 (no network, no torch)."""

import sys

from backend.utils import transformers5_env as env


def _fake_install(path):
    (path / "transformers" / "models" / "qwen3_5").mkdir(parents=True)
    return path


def test_find_installed_prefers_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(env.ENV_VAR, str(_fake_install(tmp_path / "tf5")))
    assert env.find_installed() == tmp_path / "tf5"


def test_incomplete_install_is_ignored(tmp_path, monkeypatch):
    (tmp_path / "transformers").mkdir()
    monkeypatch.setenv(env.ENV_VAR, str(tmp_path))
    monkeypatch.setattr(env, "repo_dir", lambda: tmp_path / "none")
    monkeypatch.setattr(env, "data_dir", lambda: tmp_path / "none2")
    assert env.find_installed() is None


def test_frozen_build_reports_unsupported(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert "packaged app" in env.can_install()


def test_old_python_reports_unsupported(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
    assert "3.11" in env.can_install()


def test_install_refuses_when_unsupported(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    try:
        env.install(tmp_path / "x")
    except RuntimeError as exc:
        assert "packaged app" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
