from protean.kernels import HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU
from protean.model.policy import local_kernel_edits
from protean.model.rl_layer import score, score_delta


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
