import os
import sys
import types

import pytest

from protean.hud_stream import (
    HudStreamError,
    _read_hud_user_env,
    assert_hud_auth,
    resolve_hud_api_key,
)


def test_read_hud_user_env_parses_cli_config(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nHUD_API_KEY="sk-hud-test"\nOTHER=value\n')

    assert _read_hud_user_env(env_file) == "sk-hud-test"


def test_resolve_hud_api_key_env_wins_and_syncs_settings(monkeypatch):
    fake_settings = types.SimpleNamespace(api_key=None)
    fake_module = types.SimpleNamespace(settings=fake_settings)

    monkeypatch.setenv("HUD_API_KEY", "sk-hud-env")
    monkeypatch.setitem(sys.modules, "hud", types.SimpleNamespace(settings=fake_module))
    monkeypatch.setitem(sys.modules, "hud.settings", fake_module)

    assert resolve_hud_api_key() == "sk-hud-env"
    assert fake_settings.api_key == "sk-hud-env"


def test_assert_hud_auth_uses_settings_fallback(monkeypatch):
    fake_settings = types.SimpleNamespace(api_key="sk-hud-settings")
    fake_module = types.SimpleNamespace(settings=fake_settings)

    monkeypatch.delenv("HUD_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "hud", types.SimpleNamespace(settings=fake_module))
    monkeypatch.setitem(sys.modules, "hud.settings", fake_module)

    assert_hud_auth()
    assert os.environ["HUD_API_KEY"] == "sk-hud-settings"


def test_resolve_hud_api_key_raises_when_missing(monkeypatch, tmp_path):
    fake_settings = types.SimpleNamespace(api_key=None)
    fake_module = types.SimpleNamespace(settings=fake_settings)

    monkeypatch.delenv("HUD_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "hud", types.SimpleNamespace(settings=fake_module))
    monkeypatch.setitem(sys.modules, "hud.settings", fake_module)
    monkeypatch.setattr("protean.hud_stream.Path.home", lambda: tmp_path)

    with pytest.raises(HudStreamError):
        resolve_hud_api_key()
