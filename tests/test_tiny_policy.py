import json

from protean.model.tiny_policy import TinyPolicyHead, load_trace_examples, train_tiny_policy


def _write_trace(path):
    rows = [
        {
            "event": "trial",
            "edit": "block_size_128",
            "best_score_before": [1.0, 1.0, 3],
            "delta_vs_best": {"held_out_speedup": -0.1, "reward": 0.0, "correct_held_out": 0},
        },
        {
            "event": "trial",
            "edit": "block_size_512",
            "best_score_before": [1.0, 1.0, 3],
            "delta_vs_best": {"held_out_speedup": 0.4, "reward": 0.1, "correct_held_out": 0},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_tiny_policy_has_nonzero_trainable_parameters():
    policy = TinyPolicyHead(hidden_dim=8)
    assert policy.parameter_count == 85


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
    assert payload["parameter_count"] == 85
