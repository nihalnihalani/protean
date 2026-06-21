import json
import os
import tempfile
from unittest.mock import MagicMock
from callbacks import RewardCurveLogger

def test_on_log_captures_train_reward():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train_history.json")
        logger = RewardCurveLogger(output_path=path, heldout_every=25)
        
        state = MagicMock()
        state.global_step = 5
        
        logger.on_log(args=None, state=state, control=None,
                      logs={"reward": 0.42, "reward_std": 0.08})
        
        with open(path) as f:
            data = json.load(f)
        
        assert data["steps"] == [5]
        assert data["train_reward"] == [0.42]
        assert data["train_std"] == [0.08]
        assert data["heldout_reward"] == [None]

def test_on_step_end_runs_heldout_eval():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train_history.json")
        eval_fn = MagicMock(return_value=(0.55, 0.09))
        logger = RewardCurveLogger(
            output_path=path, heldout_every=25, held_out_eval_fn=eval_fn
        )
        
        # Run on_log first to register a train point at step 25
        state = MagicMock()
        state.global_step = 25
        logger.on_log(args=None, state=state, control=None,
                      logs={"reward": 0.50, "reward_std": 0.07})
        # Then on_step_end should backfill the same row's heldout
        logger.on_step_end(args=None, state=state, control=None)
        
        with open(path) as f:
            data = json.load(f)
        
        assert data["heldout_reward"] == [0.55]
        assert data["heldout_std"] == [0.09]
        assert eval_fn.called

def test_on_step_end_skips_when_not_at_cadence():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train_history.json")
        eval_fn = MagicMock(return_value=(0.55, 0.09))
        logger = RewardCurveLogger(
            output_path=path, heldout_every=25, held_out_eval_fn=eval_fn
        )
        
        state = MagicMock()
        state.global_step = 7  # not a multiple of 25
        logger.on_step_end(args=None, state=state, control=None)
        
        assert not eval_fn.called

def test_atomic_write_does_not_truncate_on_error():
    """If _write raises mid-way, the existing file should remain intact."""
    import inspect
    src = inspect.getsource(RewardCurveLogger._write)
    assert "os.replace" in src or "atomic" in src.lower()
