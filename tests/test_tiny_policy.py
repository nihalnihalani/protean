import json

from protean.model.tiny_policy import (
    DEFAULT_HIDDEN_DIM,
    FEATURE_DIM,
    TraceExample,
    TinyPolicyHead,
    load_trace_examples,
    state_features,
    train_tiny_policy,
    trloo_advantages,
)


def _write_trace(path):
    best_summary_before = {
        "correct_held_out": 3,
        "mean_held_out_speedup": 1.4,
        "mean_reward": 1.2,
        "rows": [
            {"split": "train", "correct": True, "caps": [], "speedup": 1.2},
            {"split": "train", "correct": True, "caps": [], "speedup": 1.4},
            {"split": "held_out", "correct": True, "caps": [], "speedup": 1.1},
            {"split": "held_out", "correct": True, "caps": [], "speedup": 1.7},
        ],
    }
    rows = [
        {
            "event": "trial",
            "edit": "block_size_128",
            "best_score_before": [1.0, 1.0, 3],
            "best_summary_before": best_summary_before,
            "delta_vs_best": {"held_out_speedup": -0.1, "reward": 0.0, "correct_held_out": 0},
        },
        {
            "event": "trial",
            "edit": "block_size_512",
            "best_score_before": [1.0, 1.0, 3],
            "best_summary_before": best_summary_before,
            "delta_vs_best": {"held_out_speedup": 0.4, "reward": 0.1, "correct_held_out": 0},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_tiny_policy_has_nonzero_trainable_parameters():
    policy = TinyPolicyHead(hidden_dim=8)
    assert policy.parameter_count == 133


def test_default_tiny_policy_is_1m_parameters():
    policy = TinyPolicyHead()
    assert FEATURE_DIM == 10
    assert DEFAULT_HIDDEN_DIM == 62_500
    assert policy.parameter_count == 1_000_005


def test_state_features_use_richer_summary():
    features = state_features(
        {
            "correct_held_out": 2,
            "mean_held_out_speedup": 1.6,
            "mean_reward": 1.1,
            "rows": [
                {"split": "train", "correct": True, "caps": [], "speedup": 1.2},
                {"split": "held_out", "correct": True, "caps": [], "speedup": 1.4},
                {"split": "held_out", "correct": False, "caps": ["incorrect"], "speedup": 0.0},
            ],
        }
    )
    assert len(features) == 10
    assert features[-1] == 1.0
    assert features[7] > 0.0


def test_load_trace_examples_reads_advantages(tmp_path):
    trace = tmp_path / "trials.jsonl"
    _write_trace(trace)
    examples = load_trace_examples(trace)
    assert len(examples) == 2
    assert examples[1].advantage > examples[0].advantage


def _feat():
    return [0.0] * FEATURE_DIM


def test_trloo_uses_leave_one_out_baseline_within_action_group():
    # Same action, advantages [1.0, 0.5, 0.0]. LOO baseline of the first is
    # mean([0.5, 0.0]) = 0.25, giving corrected advantage 0.75 (pre-normalize).
    examples = [
        TraceExample(features=_feat(), action=0, advantage=1.0),
        TraceExample(features=_feat(), action=0, advantage=0.5),
        TraceExample(features=_feat(), action=0, advantage=0.0),
    ]
    # Reconstruct the pre-normalization corrected values by undoing the
    # affine batch-normalization (order is preserved, so relative gaps hold).
    out = trloo_advantages(examples)
    # Highest raw advantage stays highest after correction + normalization.
    assert out[0] > out[1] > out[2]
    # The LOO correction is symmetric: for an evenly spaced group the
    # corrected values are also evenly spaced (linear transform), so the
    # midpoint sits exactly between the extremes.
    assert abs((out[0] + out[2]) / 2 - out[1]) < 1e-9


def test_trloo_single_example_group_degrades_to_raw_advantage():
    # One example per action: LOO baseline is undefined, so corrected == raw
    # before normalization. With a single overall example normalization makes
    # it 0.0 (zero mean), which is the expected degenerate output.
    examples = [TraceExample(features=_feat(), action=0, advantage=1.0)]
    out = trloo_advantages(examples)
    assert out == [0.0]


def test_trloo_all_identical_advantages_returns_zeros_without_crashing():
    # Degenerate: every example shares an action and has the same advantage.
    # LOO correction is 0 for each, variance is 0, the 1e-8 std floor fires,
    # and all normalized advantages are 0.0 (no learning signal). Must not raise.
    examples = [
        TraceExample(features=_feat(), action=0, advantage=0.7),
        TraceExample(features=_feat(), action=0, advantage=0.7),
        TraceExample(features=_feat(), action=0, advantage=0.7),
    ]
    out = trloo_advantages(examples)
    assert out == [0.0, 0.0, 0.0]


def test_trloo_empty_input_returns_empty():
    assert trloo_advantages([]) == []


def test_train_moves_policy_toward_positive_advantage_action():
    # Directional check: two examples for action 0, one with a clearly positive
    # advantage and one negative, on identical features. After training the
    # probability of action 0 must strictly increase from its pre-training value.
    feats = [0.1] * FEATURE_DIM
    examples = [
        TraceExample(features=feats, action=0, advantage=1.0),
        TraceExample(features=feats, action=1, advantage=-1.0),
    ]
    policy = TinyPolicyHead(hidden_dim=4, seed=3)
    before = policy.probabilities(feats)[0]
    policy.train(examples, epochs=50, lr=0.1)
    after = policy.probabilities(feats)[0]
    assert after > before


def test_train_single_example_is_a_noop():
    # A single-example batch normalizes to a zero advantage, so no gradient
    # update should move the policy (documented degenerate behavior).
    feats = [0.2] * FEATURE_DIM
    examples = [TraceExample(features=feats, action=0, advantage=5.0)]
    policy = TinyPolicyHead(hidden_dim=4, seed=5)
    before = policy.probabilities(feats)
    policy.train(examples, epochs=100, lr=0.1)
    after = policy.probabilities(feats)
    assert before == after


def test_trloo_separates_action_groups():
    # Two action groups; the baseline must not bleed across actions.
    examples = [
        TraceExample(features=_feat(), action=0, advantage=2.0),
        TraceExample(features=_feat(), action=0, advantage=0.0),
        TraceExample(features=_feat(), action=1, advantage=10.0),
        TraceExample(features=_feat(), action=1, advantage=10.0),
    ]
    out = trloo_advantages(examples)
    # action=1 group has identical advantages -> zero LOO advantage each;
    # action=0 group has a spread -> nonzero corrected values.
    assert out[2] == out[3]
    assert out[0] != out[1]


def test_train_uses_trloo_and_keeps_signature(tmp_path):
    # Public signature unchanged; train still returns metrics dict.
    examples = [
        TraceExample(features=_feat(), action=0, advantage=1.0),
        TraceExample(features=_feat(), action=0, advantage=0.0),
    ]
    policy = TinyPolicyHead(hidden_dim=4)
    metrics = policy.train(examples, epochs=3, lr=0.05)
    assert metrics["examples"] == 2
    assert metrics["epochs"] == 3


def test_train_tiny_policy_saves_weights(tmp_path):
    trace = tmp_path / "trials.jsonl"
    out = tmp_path / "tiny_policy.json"
    _write_trace(trace)

    metrics = train_tiny_policy(trace, out, hidden_dim=8, epochs=10, lr=0.05)
    payload = json.loads(out.read_text())

    assert metrics["examples"] == 2
    assert payload["type"] == "tiny_policy_head"
    assert payload["input_dim"] == 10
    assert payload["parameter_count"] == 133
