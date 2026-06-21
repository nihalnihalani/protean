"""Tests for the centralized, validated settings layer (protean.config).

CPU-only. Exercises whichever backend is installed (pydantic-settings preferred,
dataclass fallback otherwise) through the shared public surface.
"""

from __future__ import annotations

import builtins
import importlib
from pathlib import Path
from typing import Any

import pytest

from protean import config
from protean.config import (
    DEFAULT_FIREWORKS_BASE_URL,
    DEFAULT_FIREWORKS_MODEL,
    ConfigError,
    ProteanSettings,
    load,
)

# Env vars the config layer reads; cleared before each test for isolation.
_MANAGED_ENV = [
    "FIREWORKS_API_KEY",
    "HUD_API_KEY",
    "TRITON_CACHE_DIR",
    "PROTEAN_LOG",
    "PROTEAN_ALLOW_DYNAMIC_MANIFEST",
    "FIREWORKS_MODEL",
    "FIREWORKS_BASE_URL",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in _MANAGED_ENV:
        monkeypatch.delenv(name, raising=False)
    # cwd to an empty dir so the pydantic backend never picks up a committed .env.
    monkeypatch.chdir(tmp_path)


def test_defaults_are_safe() -> None:
    settings = load()
    assert settings.fireworks_model == DEFAULT_FIREWORKS_MODEL
    assert settings.fireworks_base_url == DEFAULT_FIREWORKS_BASE_URL
    assert settings.triton_cache_dir == ""
    assert settings.protean_log == ""
    assert settings.protean_allow_dynamic_manifest == ""
    assert settings.fireworks_api_key_value() is None
    assert settings.hud_api_key_value() is None


def test_reads_env_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-secret-123")
    monkeypatch.setenv("HUD_API_KEY", "hud-secret-456")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/triton")
    monkeypatch.setenv("PROTEAN_LOG", "DEBUG")
    monkeypatch.setenv("PROTEAN_ALLOW_DYNAMIC_MANIFEST", "1")
    monkeypatch.setenv("FIREWORKS_MODEL", "accounts/x/models/custom")
    monkeypatch.setenv("FIREWORKS_BASE_URL", "https://example.test/v1")

    settings = load()
    assert settings.fireworks_api_key_value() == "fw-secret-123"
    assert settings.hud_api_key_value() == "hud-secret-456"
    assert settings.triton_cache_dir == "/tmp/triton"
    assert settings.protean_log == "DEBUG"
    assert settings.protean_allow_dynamic_manifest == "1"
    assert settings.fireworks_model == "accounts/x/models/custom"
    assert settings.fireworks_base_url == "https://example.test/v1"


def test_empty_api_key_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "")
    monkeypatch.setenv("HUD_API_KEY", "")
    settings = load()
    assert settings.fireworks_api_key_value() is None
    assert settings.hud_api_key_value() is None


def test_malformed_base_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_BASE_URL", "ftp://nope")
    with pytest.raises(ConfigError):
        load()


def test_empty_base_url_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_BASE_URL", "")
    with pytest.raises(ConfigError):
        load()


def test_malformed_log_level_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROTEAN_LOG", "verbose-please")
    with pytest.raises(ConfigError):
        load()


@pytest.mark.parametrize("level", ["", "0", "1", "DEBUG", "info", "Warning", "ERROR"])
def test_valid_log_levels_accepted(monkeypatch: pytest.MonkeyPatch, level: str) -> None:
    monkeypatch.setenv("PROTEAN_LOG", level)
    settings = load()
    assert settings.protean_log == level


def test_empty_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_MODEL", "")
    with pytest.raises(ConfigError):
        load()


def test_api_key_not_leaked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-super-secret")
    settings = load()
    # On the pydantic backend SecretStr masks the value; on the dataclass
    # fallback it may appear. Assert the accessor works regardless and that the
    # pydantic backend masks it.
    assert settings.fireworks_api_key_value() == "fw-super-secret"
    if config._HAVE_PYDANTIC_SETTINGS:
        assert "fw-super-secret" not in repr(settings)


def test_public_surface_exports() -> None:
    assert set(config.__all__) >= {
        "ConfigError",
        "DEFAULT_FIREWORKS_BASE_URL",
        "DEFAULT_FIREWORKS_MODEL",
        "ProteanSettings",
        "load",
    }
    assert issubclass(ConfigError, ValueError)
    assert isinstance(load(), ProteanSettings)


def test_dataclass_fallback_when_pydantic_settings_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reimport protean.config with pydantic-settings hidden to exercise the
    fallback branch. This guards against a regression where the absent-backend
    path (the scenario the fallback exists for) crashes on import or builds a
    broken ``ProteanSettings`` alias that ``isinstance`` cannot consume."""
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pydantic_settings" or name.startswith("pydantic_settings."):
            raise ImportError("pydantic_settings hidden for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    fallback = importlib.reload(config)
    try:
        assert fallback._HAVE_PYDANTIC_SETTINGS is False
        assert fallback._PydanticSettings is None
        # The alias must remain a usable type (not None) for isinstance/return use.
        settings = fallback.load()
        assert isinstance(settings, fallback.ProteanSettings)
        assert isinstance(settings, fallback._DataclassSettings)
        assert settings.fireworks_model == fallback.DEFAULT_FIREWORKS_MODEL
        assert settings.fireworks_api_key_value() is None
    finally:
        # Restore the real import machinery and reload so the rest of the suite
        # sees the pydantic-backed module again.
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(config)
