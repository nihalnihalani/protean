"""Small learned policy head trained from verifier traces.

The policy chooses the next kernel-edit action. It is intentionally dependency
free so the hackathon demo can train it anywhere the verifier runs.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

ACTION_BLOCK_SIZES = (128, 256, 512, 1024, 2048)
FEATURE_DIM = 10
DEFAULT_HIDDEN_DIM = 62_500


def action_index(edit_name: str) -> int | None:
    match = re.fullmatch(r"block_size_(\d+)", edit_name)
    if not match:
        return None
    block_size = int(match.group(1))
    if block_size not in ACTION_BLOCK_SIZES:
        return None
    return ACTION_BLOCK_SIZES.index(block_size)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def state_features(state: dict | list | tuple) -> list[float]:
    if isinstance(state, dict):
        rows = state.get("rows", [])
        train_speedups = [
            float(row.get("speedup", 0.0))
            for row in rows
            if row.get("split") == "train" and row.get("correct") and not row.get("caps")
        ]
        held_out_speedups = [
            float(row.get("speedup", 0.0))
            for row in rows
            if row.get("split") == "held_out" and row.get("correct") and not row.get("caps")
        ]
        mean_train_speedup = _mean(train_speedups)
        mean_held_out_speedup = float(state.get("mean_held_out_speedup", _mean(held_out_speedups)))
        cap_rate = sum(1 for row in rows if row.get("caps")) / len(rows) if rows else 0.0
        return [
            mean_held_out_speedup / 4.0,
            float(state.get("mean_reward", 0.0)) / 2.0,
            float(state.get("correct_held_out", 0.0)) / 3.0,
            mean_train_speedup / 4.0,
            (min(held_out_speedups) if held_out_speedups else 0.0) / 4.0,
            (max(held_out_speedups) if held_out_speedups else 0.0) / 4.0,
            (mean_held_out_speedup - mean_train_speedup) / 4.0,
            cap_rate,
            min(len(rows), 10) / 10.0,
            1.0,
        ]

    held_out_speedup, reward, correct_held_out = state
    return [
        float(held_out_speedup) / 4.0,
        float(reward) / 2.0,
        float(correct_held_out) / 3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def trloo_advantages(examples: list["TraceExample"]) -> list[float]:
    """TRLOO leave-one-out advantages, grouped by action.

    For each example, the baseline is the mean of the OTHER same-action
    advantages (Dr. Kernel arXiv:2602.05885 §3.3; Kevin arXiv:2507.11948).
    For a same-action group of size N > 1 this equals
    ``(N / (N - 1)) * (G_i - group_mean)``. Single-example groups degrade to
    the raw advantage (no other sample to baseline against). The corrected
    advantages are then batch-normalized (zero mean, unit std with a 1e-8
    floor) so the policy-gradient scale is stable across training runs.

    Normalization uses population variance (the ``len(corrected)`` denominator),
    which is the correct choice when standardizing over the *full* training
    batch rather than estimating the variance of a sample drawn from a larger
    population: the batch IS the population for this gradient step. This matches
    the standard REINFORCE/GRPO batch-normalization convention.

    Degenerate cases are intentional, total, and safe (never raise):
      * Empty input returns an empty list.
      * A single-example batch normalizes to ``[0.0]`` (the value minus its own
        mean is 0). This yields a zero policy gradient, so a one-example batch
        is effectively a training no-op -- see ``TinyPolicyHead.train``.
      * When every corrected advantage is identical (e.g. all trials returned
        the same reward), the variance is 0, the 1e-8 std floor fires, and all
        outputs are 0.0. The policy does not move, which is the desired
        behavior: an undifferentiated batch carries no learning signal.
    """

    by_action: dict[int, list[int]] = {}
    for idx, example in enumerate(examples):
        by_action.setdefault(example.action, []).append(idx)

    corrected = [0.0] * len(examples)
    for indices in by_action.values():
        group = [examples[i].advantage for i in indices]
        n = len(group)
        total = sum(group)
        for i in indices:
            g_i = examples[i].advantage
            if n > 1:
                baseline = (total - g_i) / (n - 1)
                corrected[i] = g_i - baseline
            else:
                corrected[i] = g_i

    if not corrected:
        return corrected
    mean = sum(corrected) / len(corrected)
    var = sum((value - mean) ** 2 for value in corrected) / len(corrected)
    std = max(math.sqrt(var), 1e-8)
    return [(value - mean) / std for value in corrected]


def advantage_from_delta(delta: dict) -> float:
    return (
        float(delta.get("held_out_speedup", 0.0))
        + 0.1 * float(delta.get("reward", 0.0))
        + 0.05 * float(delta.get("correct_held_out", 0.0))
    )


@dataclass(frozen=True)
class TraceExample:
    features: list[float]
    action: int
    advantage: float


class TinyPolicyHead:
    """One-hidden-layer policy head.

    Default parameter count is 1,000,005:
    input 10 -> hidden 62,500 -> 5 actions, with biases.
    """

    def __init__(
        self,
        *,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        action_dim: int = len(ACTION_BLOCK_SIZES),
        seed: int = 7,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        rng = random.Random(seed)
        scale = 0.05
        self.w1 = [[rng.uniform(-scale, scale) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        self.w2 = [[rng.uniform(-scale, scale) for _ in range(hidden_dim)] for _ in range(action_dim)]
        self.b2 = [0.0 for _ in range(action_dim)]

    @property
    def parameter_count(self) -> int:
        return self.hidden_dim * self.input_dim + self.hidden_dim + self.action_dim * self.hidden_dim + self.action_dim

    def _forward(self, features: list[float]) -> tuple[list[float], list[float]]:
        hidden = []
        for row, bias in zip(self.w1, self.b1):
            z = sum(weight * value for weight, value in zip(row, features)) + bias
            hidden.append(math.tanh(z))
        logits = []
        for row, bias in zip(self.w2, self.b2):
            logits.append(sum(weight * value for weight, value in zip(row, hidden)) + bias)
        return hidden, logits

    def probabilities(self, features: list[float]) -> list[float]:
        _, logits = self._forward(features)
        offset = max(logits)
        exp_values = [math.exp(value - offset) for value in logits]
        total = sum(exp_values)
        return [value / total for value in exp_values]

    def ranked_actions(self, features: list[float]) -> list[int]:
        probs = self.probabilities(features)
        return sorted(range(len(probs)), key=lambda idx: probs[idx], reverse=True)

    def train(self, examples: list[TraceExample], *, epochs: int = 200, lr: float = 0.05) -> dict:
        """Fit the policy head with offline batch policy gradient.

        This is OFFLINE batch RL, not on-policy REINFORCE: the per-example
        advantages are computed once from the (fixed) trace ``advantage`` fields
        via ``trloo_advantages`` *before* the epoch loop, so the baseline does
        not adapt as the policy updates. The gradient form is REINFORCE-style
        (positive advantage raises the chosen action's log-prob, negative lowers
        it), but the advantages are pre-computed labels, not freshly sampled
        returns.

        Degenerate note: a single-example batch (or any batch whose corrected
        advantages are all identical) normalizes to all-zero advantages and is
        therefore a training no-op -- no weight update moves the policy. This is
        intentional (see ``trloo_advantages``); supply at least two
        differentiated examples for the policy to learn.
        """

        if not examples:
            raise ValueError("no trace examples to train on")

        # TRLOO leave-one-out advantages: baseline each example against the
        # mean of the OTHER same-action advantages, then batch-normalize. This
        # reduces gradient variance versus using the raw per-example advantage.
        # Computed once (offline batch RL): advantages are fixed labels, not
        # recomputed per epoch as the policy moves.
        advantages = trloo_advantages(examples)

        for _ in range(epochs):
            for example, advantage in zip(examples, advantages):
                hidden, logits = self._forward(example.features)
                offset = max(logits)
                exp_values = [math.exp(value - offset) for value in logits]
                total = sum(exp_values)
                probs = [value / total for value in exp_values]

                # REINFORCE-style objective: positive advantages increase the
                # action probability; negative advantages decrease it.
                grad_logits = [advantage * prob for prob in probs]
                grad_logits[example.action] -= advantage

                old_w2 = [row[:] for row in self.w2]
                for action in range(self.action_dim):
                    for hid in range(self.hidden_dim):
                        self.w2[action][hid] -= lr * grad_logits[action] * hidden[hid]
                    self.b2[action] -= lr * grad_logits[action]

                grad_hidden = []
                for hid in range(self.hidden_dim):
                    grad = sum(grad_logits[action] * old_w2[action][hid] for action in range(self.action_dim))
                    grad_hidden.append(grad * (1.0 - hidden[hid] * hidden[hid]))

                for hid in range(self.hidden_dim):
                    for feat in range(self.input_dim):
                        self.w1[hid][feat] -= lr * grad_hidden[hid] * example.features[feat]
                    self.b1[hid] -= lr * grad_hidden[hid]

        return {
            "examples": len(examples),
            "epochs": epochs,
            "lr": lr,
            "parameter_count": self.parameter_count,
        }

    def to_dict(self) -> dict:
        return {
            "type": "tiny_policy_head",
            "actions": list(ACTION_BLOCK_SIZES),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "parameter_count": self.parameter_count,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TinyPolicyHead":
        policy = cls(
            input_dim=int(payload["input_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            action_dim=int(payload["action_dim"]),
        )
        policy.w1 = payload["w1"]
        policy.b1 = payload["b1"]
        policy.w2 = payload["w2"]
        policy.b2 = payload["b2"]
        return policy

    def save(self, path: str | Path, *, metrics: dict | None = None) -> None:
        payload = self.to_dict()
        if metrics is not None:
            payload["training"] = metrics
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TinyPolicyHead":
        return cls.from_dict(json.loads(Path(path).read_text()))


def load_trace_examples(trace_path: str | Path) -> list[TraceExample]:
    examples = []
    for line in Path(trace_path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "trial":
            continue
        idx = action_index(str(row.get("edit", "")))
        if idx is None:
            continue
        examples.append(
            TraceExample(
                features=state_features(row.get("best_summary_before", row["best_score_before"])),
                action=idx,
                advantage=advantage_from_delta(row.get("delta_vs_best", {})),
            )
        )
    return examples


def train_tiny_policy(
    trace_path: str | Path,
    out_path: str | Path,
    *,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    epochs: int = 200,
    lr: float = 0.05,
    seed: int = 7,
) -> dict:
    examples = load_trace_examples(trace_path)
    policy = TinyPolicyHead(hidden_dim=hidden_dim, seed=seed)
    metrics = policy.train(examples, epochs=epochs, lr=lr)
    metrics["trace"] = str(trace_path)
    metrics["out"] = str(out_path)
    policy.save(out_path, metrics=metrics)
    return metrics
