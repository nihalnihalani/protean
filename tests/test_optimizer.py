import json

from protean.kernels import HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU
from protean.model.policy import local_kernel_edits
from protean.model.rl_layer import score, score_delta
from protean.optimizer import run_optimization


def test_local_edits_modify_current_best_source():
    edits = list(local_kernel_edits(HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU))
    assert edits
    assert any("block_size=512" in edit.source for edit in edits)
    assert all("torch.relu(x + y)" not in edit.source for edit in edits)


def test_score_prioritizes_held_out_speed_then_reward_then_correctness():
    faster = {"mean_held_out_speedup": 2.0, "mean_reward": 1.0, "correct_held_out": 3}
    slower = {"mean_held_out_speedup": 1.5, "mean_reward": 1.3, "correct_held_out": 3}
    assert score(faster) > score(slower)


def test_score_delta_records_gain_against_current_best():
    candidate = (2.25, 1.3, 3)
    baseline = (2.0, 1.1, 2)
    assert score_delta(candidate, baseline) == {
        "held_out_speedup": 0.25,
        "reward": 0.2,
        "correct_held_out": 1,
    }


def test_optimizer_logs_rejected_trial_when_candidate_eval_crashes(tmp_path, monkeypatch):
    import protean.optimizer as optimizer

    calls = {"n": 0}

    def fake_evaluate_kernel(source, *, op, reps, warmup):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": 1.5, "mean_reward": 1.0}
        raise TypeError("grid must be a tuple")

    monkeypatch.setattr(optimizer, "evaluate_kernel", fake_evaluate_kernel)
    result = run_optimization(out_dir=tmp_path, max_rounds=1)

    rows = [(json.loads(line)) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    trial = [row for row in rows if row["event"] == "trial"][0]
    assert result["trials"] == 5
    assert result["accepted"] == 0
    assert trial["accepted"] is False
    assert trial["tokens"] == 0
    assert trial["eval_error"] == {"type": "TypeError", "message": "grid must be a tuple"}
    assert trial["summary"]["eval_error"] == trial["eval_error"]
    assert trial["score"] == [0.0, 0.0, 0]
    assert (tmp_path / trial["source_path"]).exists()
