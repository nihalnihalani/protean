import json
import sys
import types

from protean.kernels import HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU
from protean.model.policy import local_kernel_edits
from protean.model.rl_layer import score, score_delta
from protean.optimizer import evaluate_kernel, run_optimization


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
    improvements = [
        json.loads(line)
        for line in (tmp_path / "improvements_elementwise_add_relu.jsonl").read_text().splitlines()
    ]
    assert improvements[0]["event"] == "seed"
    assert improvements[1]["event"] == "trial"
    assert improvements[1]["candidate_optimizer_reward"] == 0.0
    assert improvements[1]["accepted"] is False


def test_optimizer_streams_each_trial_to_hud_when_enabled(tmp_path, monkeypatch):
    import protean.optimizer as optimizer

    calls = {"eval": 0, "stream": []}
    session = types.SimpleNamespace(job_url="https://hud.ai/jobs/session-1", name="session-1")

    def fake_evaluate_kernel(source, *, op, reps, warmup):
        calls["eval"] += 1
        if calls["eval"] == 1:
            return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": 1.5, "mean_reward": 1.0}
        return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": 1.4, "mean_reward": 0.9}

    def fake_stream_candidate_to_hud(**kwargs):
        calls["stream"].append(kwargs)
        return {
            "job_id": "session-1",
            "job_url": "https://hud.ai/jobs/session-1",
            "mean_reward": 0.5,
            "rows": [{"slug": f"{kwargs['op']}_train", "reward": 0.5}],
        }

    def fake_start_hud_stream_session(**kwargs):
        return session

    fake_hud_stream = types.ModuleType("protean.hud_stream")
    fake_hud_stream.stream_candidate_to_hud = fake_stream_candidate_to_hud
    fake_hud_stream.start_hud_stream_session = fake_start_hud_stream_session

    monkeypatch.setattr(optimizer, "evaluate_kernel", fake_evaluate_kernel)
    monkeypatch.setitem(sys.modules, "protean.hud_stream", fake_hud_stream)

    result = run_optimization(out_dir=tmp_path, max_rounds=1, stream_hud=True, hud_job_name="session-1")

    rows = [(json.loads(line)) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    trials = [row for row in rows if row["event"] == "trial"]
    assert result["stream_hud"] is True
    assert result["hud_job_url"] == "https://hud.ai/jobs/session-1"
    assert result["trials"] == 5
    assert len(calls["stream"]) == 5
    assert all(call["session"] is session for call in calls["stream"])
    assert calls["stream"][0]["summary"]["mean_reward"] == 0.9
    assert trials[0]["trial"] == 1
    assert trials[0]["op"] == "elementwise_add_relu"
    assert trials[0]["hud_stream"]["job_url"] == "https://hud.ai/jobs/session-1"
    assert trials[0]["hud_stream_error"] is None


def test_optimizer_logs_hud_stream_error_without_killing_trial(tmp_path, monkeypatch):
    import protean.optimizer as optimizer

    calls = {"eval": 0}
    session = types.SimpleNamespace(job_url="https://hud.ai/jobs/session-1", name="session-1")

    def fake_evaluate_kernel(source, *, op, reps, warmup):
        calls["eval"] += 1
        if calls["eval"] == 1:
            return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": 1.5, "mean_reward": 1.0}
        return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": 1.4, "mean_reward": 0.9}

    def fake_stream_candidate_to_hud(**kwargs):
        raise RuntimeError("hud unavailable")

    def fake_start_hud_stream_session(**kwargs):
        return session

    fake_hud_stream = types.ModuleType("protean.hud_stream")
    fake_hud_stream.stream_candidate_to_hud = fake_stream_candidate_to_hud
    fake_hud_stream.start_hud_stream_session = fake_start_hud_stream_session

    monkeypatch.setattr(optimizer, "evaluate_kernel", fake_evaluate_kernel)
    monkeypatch.setitem(sys.modules, "protean.hud_stream", fake_hud_stream)

    run_optimization(out_dir=tmp_path, max_rounds=1, stream_hud=True)

    rows = [(json.loads(line)) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    trial = [row for row in rows if row["event"] == "trial"][0]
    assert trial["hud_stream"] is None
    assert trial["hud_stream_error"] == {"type": "RuntimeError", "message": "hud unavailable"}


def test_evaluate_kernel_summary_includes_internal_optimizer_reward(monkeypatch):
    import protean.optimizer as optimizer

    def fake_grade_source(source, *, op, split, shape, reps, warmup):
        return {
            "reward": 0.5,
            "correct": True,
            "speedup": 2.0,
            "caps": [],
            "split": split,
        }

    monkeypatch.setattr(optimizer, "grade_source", fake_grade_source)
    summary = evaluate_kernel("source", reps=1, warmup=1)
    assert summary["mean_reward"] == 0.5
    assert summary["mean_optimizer_reward"] == 1.5
    assert summary["mean_held_out_optimizer_reward"] == 1.5
