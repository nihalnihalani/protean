import json

from protean.model.tiny_policy import (
    DEFAULT_HIDDEN_DIM,
    FEATURE_DIM,
    TinyPolicyHead,
    load_trace_examples,
    state_features,
    train_tiny_policy,
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
