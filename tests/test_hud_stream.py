import asyncio
import importlib.util
import os
import sys
import types

import pytest

from protean.hud_stream import (
    HudStreamError,
    _read_hud_user_env,
    _run_sync,
    _task_slugs_for_op,
    assert_hud_auth,
    resolve_hud_api_key,
)

_HUD_INSTALLED = importlib.util.find_spec("hud") is not None


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


def test_run_sync_no_running_loop():
    async def _coro():
        return 7

    assert _run_sync(_coro()) == 7


def test_run_sync_inside_running_loop_does_not_raise():
    # Simulate a sync shim being invoked from within an already-running loop
    # (e.g. a notebook). The naive asyncio.run() path would raise
    # "asyncio.run() cannot be called from a running event loop"; _run_sync
    # must instead complete the coroutine on a worker thread.
    async def _coro():
        await asyncio.sleep(0)
        return "ok"

    async def _driver():
        # Confirm a loop really is running in this thread.
        asyncio.get_running_loop()
        return _run_sync(_coro())

    assert asyncio.run(_driver()) == "ok"


def test_run_sync_propagates_exceptions_inside_running_loop():
    async def _boom():
        raise ValueError("kaboom")

    async def _driver():
        asyncio.get_running_loop()
        return _run_sync(_boom())

    with pytest.raises(ValueError, match="kaboom"):
        asyncio.run(_driver())


def test_group_documented_as_logging_only():
    import protean.hud_stream as mod

    assert "logging" in (mod.__doc__ or "").lower()
    assert "TrainingClient" in (mod.__doc__ or "")


@pytest.mark.skipif(not _HUD_INSTALLED, reason="hud extra not installed")
def test_taskset_from_file_real_sdk_contract():
    """Exercise the real hud SDK path that ``_stream_candidate_async`` relies on.

    ``_stream_candidate_async`` calls ``Taskset.from_file(env_source)`` followed
    by ``.filter(...)``. This test invokes that exact API against the installed
    SDK (offline -- ``from_file`` parses the module locally and needs no
    ``HUD_API_KEY``), so any future Taskset API version drift is caught here
    rather than silently at deploy/serve time. The env exposes one train and one
    held-out task per op (3 ops -> 6 tasks).
    """
    from hud.eval import Taskset

    taskset = Taskset.from_file("src/protean/env.py")
    slugs = {getattr(t, "slug", None) for t in taskset}
    assert len(slugs) == 6

    # The filter helper hud_stream uses must select exactly the op's two slugs.
    filtered = Taskset.from_file("src/protean/env.py").filter(_task_slugs_for_op("rmsnorm"))
    filtered_slugs = {getattr(t, "slug", None) for t in filtered}
    assert filtered_slugs == {"rmsnorm_train", "rmsnorm_held_out"}
