"""Training callbacks for the optional GRPO stretch path."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - keeps CPU/dev tests dependency-light.

    class TrainerCallback:  # type: ignore[no-redef]
        pass


class CostAbortCallback(TrainerCallback):
    """Stop overnight training when cost, time, or reward health gates fail."""

    def __init__(
        self,
        reserve_spent: float = 0.0,
        price_per_hour: float = 5.59,
        cost_limit: float = 150.0,
        time_limit_h: float = 14.0,
        review_step: int = 150,
        min_reward_at_review: float = 0.1,
        min_reward_std_at_review: float = 0.02,
    ):
        self.t_start = time.time()
        self.reserve_spent = reserve_spent
        self.price_per_hour = price_per_hour
        self.cost_limit = cost_limit
        self.time_limit_h = time_limit_h
        self.review_step = review_step
        self.min_reward_at_review = min_reward_at_review
        self.min_reward_std_at_review = min_reward_std_at_review
        self.reward_history: list[float] = []

    def _ckpt(self, trainer: Any, state: Any) -> None:
        print(f"[protean] stopping training: saving checkpoint at step {state.global_step}")
        try:
            trainer.save_model()
        except Exception as exc:
            print(f"[protean] failed to save model: {exc}")

    def _reward_flat(self, window: int = 30) -> bool:
        if len(self.reward_history) < window:
            return False
        recent = self.reward_history[-window:]
        mean_val = sum(recent) / len(recent)
        variance = sum((x - mean_val) ** 2 for x in recent) / len(recent)
        return variance**0.5 < 0.01

    def _step_review(self, state: Any) -> str | None:
        if not self.reward_history:
            return "No reward data collected by review step"
        recent = self.reward_history[-min(30, len(self.reward_history)) :]
        mean_r = sum(recent) / len(recent)
        variance = sum((x - mean_r) ** 2 for x in recent) / len(recent)
        std_r = variance**0.5
        diagnostics: list[str] = []
        if mean_r < self.min_reward_at_review:
            diagnostics.append(f"mean_reward={mean_r:.4f} < {self.min_reward_at_review}")
        if std_r < self.min_reward_std_at_review:
            diagnostics.append(f"reward_std={std_r:.4f} < {self.min_reward_std_at_review}")
        return "; ".join(diagnostics) if diagnostics else None

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        elapsed_h = (time.time() - self.t_start) / 3600.0
        spent = self.reserve_spent + elapsed_h * self.price_per_hour

        for log in reversed(getattr(state, "log_history", []) or []):
            for key in ("reward", "train_reward", "train/reward", "rewards/mean"):
                if key in log:
                    self.reward_history.append(float(log[key]))
                    break
            else:
                continue
            break

        trainer = kwargs.get("trainer")
        should_stop = False
        if spent > self.cost_limit:
            print(f"[protean] cost limit reached: spent ${spent:.2f}")
            should_stop = True
        if elapsed_h > self.time_limit_h:
            print(f"[protean] time limit reached: elapsed {elapsed_h:.2f} hours")
            should_stop = True
        if getattr(state, "global_step", 0) >= 50 and self._reward_flat(window=30):
            print("[protean] reward curve flat. stopping early.")
            should_stop = True
        if getattr(state, "global_step", 0) == 75 and self.reward_history:
            recent = self.reward_history[-min(15, len(self.reward_history)) :]
            mean_r = sum(recent) / len(recent)
            if mean_r < 0.05:
                print(f"[protean] Step-75 early warning: mean_reward={mean_r:.4f} < 0.05")
        if getattr(state, "global_step", 0) == self.review_step:
            diag = self._step_review(state)
            if diag:
                print(f"[protean] Step-{self.review_step} review FAILED: {diag}")
                should_stop = True
            else:
                mean_r = sum(self.reward_history[-30:]) / min(30, len(self.reward_history))
                print(f"[protean] Step-{self.review_step} review PASSED: mean_reward={mean_r:.4f}")

        if should_stop:
            if trainer is not None:
                self._ckpt(trainer, state)
            control.should_training_stop = True


class RewardCurveLogger(TrainerCallback):
    """Serialize train and held-out reward history for real curve plots."""

    def __init__(
        self,
        output_path: str,
        heldout_every: int = 25,
        heldout_tasks_per_op: int = 4,
        held_out_eval_fn: Callable[..., tuple[float, float]] | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ):
        self.output_path = output_path
        self.heldout_every = heldout_every
        self.heldout_tasks_per_op = heldout_tasks_per_op
        self.held_out_eval_fn = held_out_eval_fn
        self._history: dict[str, Any] = {
            "steps": [],
            "train_reward": [],
            "train_std": [],
            "heldout_reward": [],
            "heldout_std": [],
            "config": config_snapshot or {},
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        logs = logs or {}
        _reward_keys = ("reward", "train_reward", "train/reward", "rewards/mean")
        _std_keys = ("reward_std", "train_reward_std", "train/reward_std", "rewards/std")
        train_reward = next((logs[k] for k in _reward_keys if k in logs), None)
        train_std = next((logs[k] for k in _std_keys if k in logs), None)
        if train_reward is None:
            return
        self._history["steps"].append(int(state.global_step))
        self._history["train_reward"].append(float(train_reward))
        self._history["train_std"].append(float(train_std) if train_std is not None else 0.0)
        self._history["heldout_reward"].append(None)
        self._history["heldout_std"].append(None)
        self._write()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        step = int(getattr(state, "global_step", 0))
        if step == 0 or step % self.heldout_every != 0 or self.held_out_eval_fn is None:
            return
        try:
            heldout_mean, heldout_std = self.held_out_eval_fn(step=step, n_per_op=self.heldout_tasks_per_op)
        except Exception as exc:
            print(f"[protean] held-out eval at step {step} failed: {type(exc).__name__}: {exc}")
            return
        if self._history["steps"] and self._history["steps"][-1] == step:
            idx = len(self._history["steps"]) - 1
            self._history["heldout_reward"][idx] = float(heldout_mean)
            self._history["heldout_std"][idx] = float(heldout_std)
        else:
            self._history["steps"].append(step)
            self._history["train_reward"].append(None)
            self._history["train_std"].append(None)
            self._history["heldout_reward"].append(float(heldout_mean))
            self._history["heldout_std"].append(float(heldout_std))
        self._write()
        print(f"[protean] held-out eval step={step}: mean={heldout_mean:.3f}, std={heldout_std:.3f}")

    def _write(self) -> None:
        tmp = self.output_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)
        os.replace(tmp, self.output_path)
