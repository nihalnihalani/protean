import pytest

from protean.env import (
    EXPECTED_TEMPLATE_IDS,
    HUD_TASKS,
    assert_templates_registered,
    grade_hud_source,
    hud_prompt,
    registered_template_ids,
    task_metadata,
)
from protean.grader import to_eval_result
from protean.kernels import PYTORCH_PASSTHROUGH


def test_hud_exposes_six_task_ids():
    ids = {task["id"] for task in HUD_TASKS}
    assert ids == {
        "elementwise_add_relu_train",
        "elementwise_add_relu_held_out",
        "rmsnorm_train",
        "rmsnorm_held_out",
        "softmax_rows_train",
        "softmax_rows_held_out",
    }


def test_hud_prompt_includes_metadata():
    prompt = hud_prompt("rmsnorm", "held_out", 1536)
    assert "op: rmsnorm" in prompt
    assert "split: held_out" in prompt
    assert "shape: 1536" in prompt


def test_hud_grade_metadata_contains_reward_fields():
    grade = grade_hud_source(PYTORCH_PASSTHROUGH, op="elementwise_add_relu", split="held_out", shape=1536)
    assert grade["reward"] == 0.0
    assert grade["hud"]["op"] == "elementwise_add_relu"
    assert grade["hud"]["split"] == "held_out"
    assert grade["hud"]["shape"] == 1536
    assert "caps" in grade["hud"]


def test_task_metadata_lists_prompt_paths():
    rows = task_metadata()
    assert len(rows) == 6
    assert all(row["prompt_path"].endswith("prompt.md") for row in rows)


def test_hud_eval_result_normalizes_reward_and_keeps_raw_score():
    result = to_eval_result(
        {
            "reward": 1.3,
            "correct": True,
            "speedup_score": 0.42,
            "split": "held_out",
            "caps": [],
        }
    )
    assert result.reward == 0.65
    assert result.info["protean_reward_raw"] == 1.3
    assert {subscore.name for subscore in result.subscores} == {
        "hud_reward",
        "correctness",
        "speedup",
        "held_out",
        "anti_hack",
        "compile_success",
    }
    assert all(0.0 <= subscore.value <= 1.0 for subscore in result.subscores)


def test_hud_registration_uses_name_api():
    import protean.env as env_mod

    calls = {"env_names": [], "template_ids": []}

    class FakeEnvironment:
        def __init__(self, *, name):
            calls["env_names"].append(name)

        def template(self, *, id):
            calls["template_ids"].append(id)

            def decorator(fn):
                return fn

            return decorator

    original_environment = env_mod.Environment
    original_env = env_mod.env
    try:
        env_mod.Environment = FakeEnvironment
        env_mod.env = env_mod._make_env()
        decorator_add = env_mod._template("elementwise_add_relu")
        decorator_rms = env_mod._template("rmsnorm")
        decorator_softmax = env_mod._template("softmax_rows")

        def placeholder():
            return None

        assert decorator_add(placeholder) is placeholder
        assert decorator_rms(placeholder) is placeholder
        assert decorator_softmax(placeholder) is placeholder
    finally:
        env_mod.Environment = original_environment
        env_mod.env = original_env

    assert calls["env_names"] == ["protean"]
    assert calls["template_ids"] == ["elementwise_add_relu", "rmsnorm", "softmax_rows"]


def test_expected_template_ids_match_three_ops():
    assert EXPECTED_TEMPLATE_IDS == ("elementwise_add_relu", "rmsnorm", "softmax_rows")


def test_registered_template_ids_when_hud_present():
    import protean.env as env_mod

    if env_mod.env is None:
        pytest.skip("hud not installed; registration is a no-op without the hud extra")

    registered = set(registered_template_ids())
    assert set(EXPECTED_TEMPLATE_IDS) <= registered
    # All three expected templates registered, none dropped.
    assert len(set(EXPECTED_TEMPLATE_IDS) & registered) == 3


def test_assert_templates_registered_passes_when_hud_present():
    import protean.env as env_mod

    if env_mod.env is None:
        pytest.skip("hud not installed; serve-time guard is exercised separately")

    # Should not raise: the live env registered all expected templates at import.
    assert_templates_registered()


def test_templates_registered_immediately_after_import():
    """Pin the eager-registration timing the import-time serve guard relies on.

    The @_template decorators run at module import, so registered_template_ids()
    must return all three expected ids immediately after import -- before any serve
    hook runs. If a future HUD SDK switches to lazy registration, this test fails,
    flagging that the import-time assert_templates_registered() guard must move into
    @env.initialize only.
    """

    import protean.env as env_mod

    if env_mod.env is None:
        pytest.skip("hud not installed; eager registration is a no-op without the hud extra")

    assert set(registered_template_ids()) == set(EXPECTED_TEMPLATE_IDS)


def test_registered_templates_are_not_dict_stubs():
    """When hud is present, the registered templates must be real HUD objects.

    Guards against vacuously-passing registration where the hud-absent dict stubs
    (env.py's else-branch) leak into the registered set: a dict-keyed registry of
    the right ids would still satisfy the id check, so assert the env itself is a
    real Environment (not None) and that the live registry is non-empty.
    """

    import protean.env as env_mod

    if env_mod.env is None:
        pytest.skip("hud not installed; registration is a no-op without the hud extra")

    assert not isinstance(env_mod.env, dict)
    assert len(registered_template_ids()) >= len(EXPECTED_TEMPLATE_IDS)


def test_assert_templates_registered_fails_loud_when_env_missing(monkeypatch):
    import protean.env as env_mod

    # Simulate a deploy image where the hud import failed: env is None. The guard
    # must raise rather than let a zero-template env be served silently.
    monkeypatch.setattr(env_mod, "env", None)
    assert registered_template_ids() == ()
    with pytest.raises(RuntimeError, match="not importable"):
        env_mod.assert_templates_registered()


def test_assert_templates_registered_fails_loud_when_templates_missing(monkeypatch):
    import protean.env as env_mod

    class _EmptyEnv:
        tasks: dict[str, object] = {}
        templates: dict[str, object] = {}

    monkeypatch.setattr(env_mod, "env", _EmptyEnv())
    assert env_mod.registered_template_ids() == ()
    with pytest.raises(RuntimeError, match="registration"):
        env_mod.assert_templates_registered()
