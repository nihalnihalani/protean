import json
import sys
import types

import pytest

from protean.kernels import HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU, seed_kernel_for
from protean.model.policy import (
    block_size_is_tunable,
    config_action_space,
    effective_action_space,
    local_kernel_edits,
    make_config_edit,
)
from protean.model.rl_layer import score, score_delta
from protean.optimizer import BanditSearch, bandit_reward, run_optimization


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


# --- Config-space edit helpers (KernelBand joint launch-config search) -------


def test_config_action_space_is_full_cartesian_product():
    arms = config_action_space()
    # 5 block sizes x 3 num_warps x 3 num_stages.
    assert len(arms) == 45
    assert len(set(arms)) == 45
    assert (1024, 4, 2) in arms


def test_make_config_edit_injects_block_warps_and_stages():
    edit = make_config_edit(seed_kernel_for("elementwise_add_relu"), 256, 8, 3)
    assert edit.name == "config_b256_w8_s3"
    assert edit.policy == "bandit_config"
    assert "block_size=256" in edit.source
    assert "num_warps=8" in edit.source
    assert "num_stages=3" in edit.source
    # The grid divisor must follow the block size.
    assert "triton.cdiv(n_elements, 256)" in edit.source


def test_make_config_edit_is_idempotent_across_rounds():
    once = make_config_edit(seed_kernel_for("elementwise_add_relu"), 512, 4, 2)
    twice = make_config_edit(once.source, 128, 16, 4)
    # Re-editing must not accumulate stale meta kwargs.
    assert twice.source.count("num_warps=") == 1
    assert twice.source.count("num_stages=") == 1
    assert "num_warps=16" in twice.source
    assert "num_stages=4" in twice.source
    assert "num_warps=4" not in twice.source


def test_make_config_edit_on_rmsnorm_injects_warps_and_stages():
    # rmsnorm derives block_size from the data (_next_power_of_2), so the literal
    # block_size axis is intentionally inert -- but warps/stages must still land
    # on the actual kernel launch.
    edit = make_config_edit(seed_kernel_for("rmsnorm"), 256, 8, 3)
    assert "num_warps=8" in edit.source
    assert "num_stages=3" in edit.source
    # The meta kwargs must sit on the kernel launch, not on a helper call.
    assert "_rmsnorm_kernel[(1,)]" in edit.source
    assert "num_warps=8, num_stages=3)" in edit.source
    # Data-derived block size is preserved (forcing a literal would be incorrect).
    assert "_next_power_of_2(n_elements)" in edit.source
    # Exactly one injection -- no duplication into the helper definition.
    assert edit.source.count("num_warps=") == 1
    assert edit.source.count("num_stages=") == 1


def test_make_config_edit_on_softmax_rows_injects_warps_and_stages():
    edit = make_config_edit(seed_kernel_for("softmax_rows"), 512, 16, 4)
    assert "num_warps=16" in edit.source
    assert "num_stages=4" in edit.source
    assert "_softmax_rows_kernel[(n_rows,)]" in edit.source
    assert "num_warps=16, num_stages=4)" in edit.source
    assert "_next_power_of_2(n_cols)" in edit.source
    assert edit.source.count("num_warps=") == 1
    assert edit.source.count("num_stages=") == 1


def test_block_size_is_tunable_only_for_elementwise():
    assert block_size_is_tunable(seed_kernel_for("elementwise_add_relu")) is True
    assert block_size_is_tunable(seed_kernel_for("rmsnorm")) is False
    assert block_size_is_tunable(seed_kernel_for("softmax_rows")) is False


def test_effective_action_space_collapses_block_size_for_dynamic_block_ops():
    # Elementwise keeps the full 45-arm Cartesian product.
    full = effective_action_space(seed_kernel_for("elementwise_add_relu"))
    assert len(full) == 45
    # rmsnorm/softmax_rows collapse block_size -> 3 warps x 3 stages = 9 distinct
    # arms, all sharing the same pinned block size.
    for op in ("rmsnorm", "softmax_rows"):
        pruned = effective_action_space(seed_kernel_for(op))
        assert len(pruned) == 9
        assert len({arm[0] for arm in pruned}) == 1  # one block size
        assert len(set(pruned)) == 9  # distinct (warps, stages) pairs


# --- BanditSearch UCB1 over the joint config space --------------------------


def test_bandit_visits_every_arm_before_repeating():
    bandit = BanditSearch(arms=[(128, 4, 2), (256, 8, 3), (512, 16, 4)], seed=1)
    seen = set()
    for _ in range(3):
        arm = bandit.select()
        assert arm not in seen
        seen.add(arm)
        bandit.update(arm, 0.0)
    assert seen == {(128, 4, 2), (256, 8, 3), (512, 16, 4)}


def test_bandit_concentrates_on_high_reward_arm():
    arms = [(128, 4, 2), (256, 8, 3), (512, 16, 4)]
    bandit = BanditSearch(arms=arms, seed=3)
    winner = (256, 8, 3)
    rewards = {(128, 4, 2): 0.2, (256, 8, 3): 3.0, (512, 16, 4): 0.1}
    for _ in range(120):
        arm = bandit.select()
        bandit.update(arm, rewards[arm])
    # UCB should pull the best arm far more than the losers.
    assert bandit.best_arm() == winner
    assert bandit.counts[winner] > sum(bandit.counts[a] for a in arms if a != winner)


def test_bandit_state_is_json_serialisable():
    bandit = BanditSearch(arms=[(128, 4, 2), (256, 8, 3)], seed=0)
    bandit.update((128, 4, 2), 1.5)
    state = bandit.state()
    json.dumps(state)  # must not raise
    assert state["total"] == 1
    assert any(arm["count"] == 1 for arm in state["arms"])


def test_bandit_requires_at_least_one_arm():
    with pytest.raises(ValueError):
        BanditSearch(arms=[])


def test_bandit_reward_prefers_held_out_speed():
    fast = {"mean_held_out_speedup": 2.0, "mean_reward": 0.5}
    slow = {"mean_held_out_speedup": 1.0, "mean_reward": 0.5}
    assert bandit_reward(fast) > bandit_reward(slow)
    assert bandit_reward({}) == 0.0


# --- run_optimization with the opt-in bandit edit policy --------------------


def _bandit_fake_eval_factory(calls):
    """Return a fake evaluate_kernel that rewards block_size=512 the most."""

    def fake_evaluate_kernel(source, *, op, reps, warmup):
        calls["n"] += 1
        if calls["n"] == 1:
            # Seed evaluation.
            return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": 1.0, "mean_reward": 0.5}
        speedup = 3.0 if "block_size=512" in source else 1.1
        return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": speedup, "mean_reward": 0.5}

    return fake_evaluate_kernel


def test_bandit_policy_runs_one_trial_per_round_and_logs_arm(tmp_path, monkeypatch):
    import protean.optimizer as optimizer

    calls = {"n": 0}
    monkeypatch.setattr(optimizer, "evaluate_kernel", _bandit_fake_eval_factory(calls))

    result = run_optimization(out_dir=tmp_path, max_rounds=20, edit_policy="bandit", bandit_seed=2)

    assert result["edit_policy"] == "bandit"
    # One candidate per round (bandit selects a single arm).
    assert result["trials"] == 20
    assert result["rounds_run"] == 20
    assert result["bandit"] is not None
    assert result["bandit_best_arm"]["block_size"] == 512

    rows = [json.loads(line) for line in (tmp_path / "trials.jsonl").read_text().splitlines()]
    trials = [row for row in rows if row["event"] == "trial"]
    assert all(row["bandit_arm"] is not None for row in trials)
    assert all(row["policy"] == "bandit_config" for row in trials)
    # The winning arm should be pulled more than any individual loser.
    block_512 = sum(1 for row in trials if row["bandit_arm"]["block_size"] == 512)
    block_other = sum(1 for row in trials if row["bandit_arm"]["block_size"] != 512)
    assert block_512 > 0
    assert block_512 >= block_other / 4


def test_bandit_anytime_stopping_halts_after_patience(tmp_path, monkeypatch):
    import protean.optimizer as optimizer

    calls = {"n": 0}

    def fake_evaluate_kernel(source, *, op, reps, warmup):
        calls["n"] += 1
        # Seed is strong; nothing ever improves on it, so no round improves.
        if calls["n"] == 1:
            return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": 5.0, "mean_reward": 1.0}
        return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": 1.0, "mean_reward": 0.5}

    monkeypatch.setattr(optimizer, "evaluate_kernel", fake_evaluate_kernel)

    result = run_optimization(
        out_dir=tmp_path,
        max_rounds=50,
        edit_policy="bandit",
        bandit_patience=3,
        bandit_seed=0,
        bandit_arms=[(128, 4, 2), (256, 8, 3)],
    )

    # Two arms are visited in rounds 1-2 (exploration completes at round 2);
    # the patience counter then accumulates over rounds 2,3,4 and fires at 3.
    assert result["stopped_early"] is True
    assert result["rounds_run"] == 4
    assert result["trials"] == 4
    assert result["accepted"] == 0


def test_bandit_patience_does_not_truncate_non_bandit_policies(tmp_path, monkeypatch):
    # bandit_patience must be a no-op for the deterministic sweep: a user who
    # sets it via config while running edit_policy='local' must not get a
    # silently truncated run.
    import protean.optimizer as optimizer

    calls = {"n": 0}

    def fake_evaluate_kernel(source, *, op, reps, warmup):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": 5.0, "mean_reward": 1.0}
        return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": 1.0, "mean_reward": 0.5}

    monkeypatch.setattr(optimizer, "evaluate_kernel", fake_evaluate_kernel)
    result = run_optimization(
        out_dir=tmp_path, max_rounds=3, edit_policy="local", bandit_patience=1
    )

    assert result["edit_policy"] == "local"
    assert result["stopped_early"] is False
    assert result["rounds_run"] == 3
    # 3 rounds x 5 deterministic block-size edits.
    assert result["trials"] == 15


def test_bandit_does_not_stop_before_all_arms_visited(tmp_path, monkeypatch):
    # Even if no arm beats the seed, anytime stopping must wait until UCB has
    # pulled every arm at least once -- otherwise it aborts while unexplored
    # arms could still win.
    import protean.optimizer as optimizer

    calls = {"n": 0}

    def fake_evaluate_kernel(source, *, op, reps, warmup):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": 5.0, "mean_reward": 1.0}
        return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": 1.0, "mean_reward": 0.5}

    monkeypatch.setattr(optimizer, "evaluate_kernel", fake_evaluate_kernel)
    result = run_optimization(
        out_dir=tmp_path,
        max_rounds=20,
        edit_policy="bandit",
        bandit_patience=1,
        bandit_seed=0,
        bandit_arms=[(128, 4, 2), (256, 8, 3), (512, 16, 4), (1024, 4, 2)],
    )

    # patience=1 with 4 arms: the counter stays 0 through the forced-exploration
    # rounds; exploration completes when the 4th arm is pulled (round 4), which
    # is also a non-improving round, so the counter hits 1 and stops at round 4.
    assert result["stopped_early"] is True
    assert result["rounds_run"] == 4
    # Every arm visited at least once before stopping.
    assert all(arm["count"] >= 1 for arm in result["bandit"]["arms"])


def test_default_local_policy_unchanged_by_bandit_additions(tmp_path, monkeypatch):
    import protean.optimizer as optimizer

    calls = {"n": 0}

    def fake_evaluate_kernel(source, *, op, reps, warmup):
        calls["n"] += 1
        return {"rows": [], "correct_held_out": 3, "mean_held_out_speedup": 1.0, "mean_reward": 0.5}

    monkeypatch.setattr(optimizer, "evaluate_kernel", fake_evaluate_kernel)
    result = run_optimization(out_dir=tmp_path, max_rounds=1)

    # Deterministic sweep path is the default and still produces 5 block-size trials.
    assert result["edit_policy"] == "local"
    assert result["trials"] == 5
    assert result["bandit"] is None
    assert result["stopped_early"] is False
