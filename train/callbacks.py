"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# callbacks.py
import time
from transformers import TrainerCallback

class CostAbortCallback(TrainerCallback):
    def __init__(self, reserve_spent=0.0, price_per_hour=5.59, cost_limit=150.0, time_limit_h=14.0,
                 review_step=150, min_reward_at_review=0.1, min_reward_std_at_review=0.02):
        self.t_start = time.time()
        self.reserve_spent = reserve_spent
        self.price_per_hour = price_per_hour
        self.cost_limit = cost_limit
        self.time_limit_h = time_limit_h
        self.reward_history = []
        # Step-150 review milestone (Step 10)
        self.review_step = review_step
        self.min_reward_at_review = min_reward_at_review
        self.min_reward_std_at_review = min_reward_std_at_review

    def _ckpt(self, trainer, state):
        print(f"Aborting training: saving checkpoint at step {state.global_step}...")
        try:
            trainer.save_model()
        except Exception as e:
            print(f"Failed to save model: {e}")

    def _reward_flat(self, window=30) -> bool:
        """Detect a flatlined reward curve. Window tightened from 40→30 (Step 10)."""
        if len(self.reward_history) < window:
            return False
        recent = self.reward_history[-window:]
        mean_val = sum(recent) / len(recent)
        variance = sum((x - mean_val) ** 2 for x in recent) / len(recent)
        std_dev = variance ** 0.5
        return std_dev < 0.01

    def _step_review(self, state) -> str | None:
        """Step-150 formal review gate (Step 10, BUILD_CHECKLIST §Block 4).
        
        At the review milestone, check that the model has shown meaningful
        learning. If the mean reward is below threshold or the curve has no
        variance, there's no point burning more GPU hours.
        
        Returns None if OK, or a diagnostic string if training should stop.
        """
        if not self.reward_history:
            return "No reward data collected by review step"
        
        recent = self.reward_history[-min(30, len(self.reward_history)):]
        mean_r = sum(recent) / len(recent)
        variance = sum((x - mean_r) ** 2 for x in recent) / len(recent)
        std_r = variance ** 0.5
        
        diagnostics = []
        if mean_r < self.min_reward_at_review:
            diagnostics.append(
                f"mean_reward={mean_r:.4f} < {self.min_reward_at_review} — "
                f"model has not learned to produce rewarded kernels"
            )
        if std_r < self.min_reward_std_at_review:
            diagnostics.append(
                f"reward_std={std_r:.4f} < {self.min_reward_std_at_review} — "
                f"no learning signal variance (flatlined)"
            )
        
        if diagnostics:
            return "; ".join(diagnostics)
        return None

    def on_step_end(self, args, state, control, **kw):
        elapsed_h = (time.time() - self.t_start) / 3600.0
        spent = self.reserve_spent + elapsed_h * self.price_per_hour
        
        # Track logged reward
        if state.log_history:
            for log in reversed(state.log_history):
                if "reward" in log:
                    self.reward_history.append(log["reward"])
                    break
                elif "train_reward" in log:
                    self.reward_history.append(log["train_reward"])
                    break
                    
        trainer = kw.get("trainer")
        if spent > self.cost_limit:
            print(f"Cost limit reached: spent ${spent:.2f}")
            if trainer:
                self._ckpt(trainer, state)
            control.should_training_stop = True
            
        if elapsed_h > self.time_limit_h:
            print(f"Time limit reached: elapsed {elapsed_h:.2f} hours")
            if trainer:
                self._ckpt(trainer, state)
            control.should_training_stop = True
            
        if state.global_step >= 50 and self._reward_flat(window=30):
            print("Reward curve flat. Stopping early.")
            if trainer:
                self._ckpt(trainer, state)
            control.should_training_stop = True
        
        # Step-75 early warning (halfway sanity check)
        if state.global_step == 75 and self.reward_history:
            recent = self.reward_history[-min(15, len(self.reward_history)):]
            mean_r = sum(recent) / len(recent)
            if mean_r < 0.05:
                print(f"[protean] ⚠ Step-75 early warning: mean_reward={mean_r:.4f} < 0.05")
                print(f"[protean] ⚠ Model may not be learning. Will re-check at step {self.review_step}.")
        
        # Step-150 formal review milestone
        if state.global_step == self.review_step:
            diag = self._step_review(state)
            if diag:
                print(f"[protean] ✗ Step-{self.review_step} review FAILED: {diag}")
                print(f"[protean] Halting training to avoid burning GPU hours on a non-learning model.")
                if trainer:
                    self._ckpt(trainer, state)
                control.should_training_stop = True
            else:
                mean_r = sum(self.reward_history[-30:]) / min(30, len(self.reward_history))
                print(f"[protean] ✓ Step-{self.review_step} review PASSED: mean_reward={mean_r:.4f}")
                print(f"[protean] Training may continue beyond step {self.review_step}.")

import json
import os
from typing import Optional

class RewardCurveLogger(TrainerCallback):
    """Captures train and held-out reward curves to disk for the money slide.
    
    Train rewards: scraped from trl's log_history on every logging step.
    Held-out rewards: re-evaluated against held-out shapes every `heldout_every` steps.
    
    Output schema (outputs/train_history.json):
        {
          "steps": [0, 5, 10, ...],
          "train_reward": [0.35, 0.42, ...],   # mean across the group
          "train_std": [0.08, 0.07, ...],       # std across the group
          "heldout_reward": [null, null, 0.31, ...],  # null when no eval at that step
          "heldout_std": [null, null, 0.09, ...],
          "config": {...}                        # snapshot of resolved GRPOConfig
        }
    
    Held-out null entries are important — the plotter handles them by skipping/
    interpolating instead of treating them as zero rewards.
    """
    
    def __init__(
        self,
        output_path: str,
        heldout_every: int = 25,
        heldout_tasks_per_op: int = 4,
        held_out_eval_fn: Optional[callable] = None,
        config_snapshot: Optional[dict] = None,
    ):
        self.output_path = output_path
        self.heldout_every = heldout_every
        self.heldout_tasks_per_op = heldout_tasks_per_op
        self.held_out_eval_fn = held_out_eval_fn  # injected to keep callback testable
        self.config_snapshot = config_snapshot or {}
        
        self._history = {
            "steps": [],
            "train_reward": [],
            "train_std": [],
            "heldout_reward": [],
            "heldout_std": [],
            "config": self.config_snapshot,
        }
        
        # Ensure the output dir exists
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called by trl every `logging_steps`. Capture train reward stats."""
        if logs is None:
            return
        
        # trl logs reward stats with keys like "reward", "reward_std", or "train/reward".
        # We try both common forms; the trl version determines which.
        train_reward = None
        train_std = None
        for k in ("reward", "train_reward", "train/reward", "rewards/mean"):
            if k in logs:
                train_reward = logs[k]
                break
        for k in ("reward_std", "train_reward_std", "train/reward_std", "rewards/std"):
            if k in logs:
                train_std = logs[k]
                break
        
        if train_reward is None:
            # Don't record a point if we couldn't find a reward. Some log calls
            # are loss-only or eval-only and don't include reward stats.
            return
        
        step = state.global_step
        self._history["steps"].append(step)
        self._history["train_reward"].append(float(train_reward))
        self._history["train_std"].append(float(train_std) if train_std is not None else 0.0)
        # Held-out is initially null at every step; we backfill in on_step_end.
        self._history["heldout_reward"].append(None)
        self._history["heldout_std"].append(None)
        
        self._write()
    
    def on_step_end(self, args, state, control, **kwargs):
        """Run held-out eval every `heldout_every` steps and backfill the curve."""
        step = state.global_step
        if step == 0 or step % self.heldout_every != 0:
            return
        if self.held_out_eval_fn is None:
            # No eval function injected — skip silently (test mode).
            return
        
        try:
            heldout_mean, heldout_std = self.held_out_eval_fn(
                step=step, n_per_op=self.heldout_tasks_per_op
            )
        except Exception as e:
            print(f"[protean] Held-out eval at step {step} failed: {type(e).__name__}: {e}")
            return
        
        # Find the most recent train-reward entry to attach the held-out point to.
        # If no train entry exists yet at this step, append a new point with train=null.
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
        print(f"[protean] Held-out eval step={step}: mean={heldout_mean:.3f}, std={heldout_std:.3f}")
    
    def _write(self):
        """Atomic write: tmp file + os.replace."""
        tmp = self.output_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._history, f, indent=2)
        os.replace(tmp, self.output_path)

