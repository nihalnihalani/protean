"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# callbacks.py
import time
from transformers import TrainerCallback

class CostAbortCallback(TrainerCallback):
    def __init__(self, reserve_spent=0.0, price_per_hour=5.59, cost_limit=150.0, time_limit_h=14.0):
        self.t_start = time.time()
        self.reserve_spent = reserve_spent
        self.price_per_hour = price_per_hour
        self.cost_limit = cost_limit
        self.time_limit_h = time_limit_h
        self.reward_history = []

    def _ckpt(self, trainer, state):
        print(f"Aborting training: saving checkpoint at step {state.global_step}...")
        try:
            trainer.save_model()
        except Exception as e:
            print(f"Failed to save model: {e}")

    def _reward_flat(self, window=40) -> bool:
        if len(self.reward_history) < window:
            return False
        recent = self.reward_history[-window:]
        mean_val = sum(recent) / len(recent)
        variance = sum((x - mean_val) ** 2 for x in recent) / len(recent)
        std_dev = variance ** 0.5
        return std_dev < 0.01

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
            
        if state.global_step >= 50 and self._reward_flat(window=40):
            print("Reward curve flat. Stopping early.")
            if trainer:
                self._ckpt(trainer, state)
            control.should_training_stop = True
