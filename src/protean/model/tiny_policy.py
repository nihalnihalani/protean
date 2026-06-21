"""Tiny learned policy head trained from verifier traces.

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
DEFAULT_HIDDEN_DIM = 12_800


def action_index(edit_name: str) -> int | None:
    match = re.fullmatch(r"block_size_(\d+)", edit_name)
    if not match:
        return None
    block_size = int(match.group(1))
    if block_size not in ACTION_BLOCK_SIZES:
        return None
    return ACTION_BLOCK_SIZES.index(block_size)


def state_features(best_score: list | tuple) -> list[float]:
    held_out_speedup, reward, correct_held_out = best_score
    return [
        float(held_out_speedup) / 4.0,
        float(reward) / 2.0,
        float(correct_held_out) / 3.0,
        1.0,
    ]


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

    Default parameter count is 128,005:
    input 4 -> hidden 12,800 -> 5 actions, with biases.
    """

    def __init__(
        self,
        *,
        input_dim: int = 4,
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
        if not examples:
            raise ValueError("no trace examples to train on")

        for _ in range(epochs):
            for example in examples:
                hidden, logits = self._forward(example.features)
                offset = max(logits)
                exp_values = [math.exp(value - offset) for value in logits]
                total = sum(exp_values)
                probs = [value / total for value in exp_values]

                # REINFORCE-style objective: positive advantages increase the
                # action probability; negative advantages decrease it.
                grad_logits = [example.advantage * prob for prob in probs]
                grad_logits[example.action] -= example.advantage

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
                features=state_features(row["best_score_before"]),
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
