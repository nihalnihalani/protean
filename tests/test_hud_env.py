from protean.env import HUD_TASKS, grade_hud_source, hud_prompt, task_metadata
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

        def placeholder():
            return None

        assert decorator_add(placeholder) is placeholder
        assert decorator_rms(placeholder) is placeholder
    finally:
        env_mod.Environment = original_environment
        env_mod.env = original_env

    assert calls["env_names"] == ["protean"]
    assert calls["template_ids"] == ["elementwise_add_relu", "rmsnorm"]
