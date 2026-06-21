import json

from train.callbacks import RewardCurveLogger


class State:
    def __init__(self, step: int):
        self.global_step = step


def test_reward_curve_logger_writes_train_history(tmp_path):
    path = tmp_path / "train_history.json"
    logger = RewardCurveLogger(str(path), heldout_every=25)
    logger.on_log(None, State(5), None, logs={"reward": 0.42, "reward_std": 0.08})

    data = json.loads(path.read_text())
    assert data["steps"] == [5]
    assert data["train_reward"] == [0.42]
    assert data["train_std"] == [0.08]
    assert data["heldout_reward"] == [None]


def test_reward_curve_logger_backfills_heldout_eval(tmp_path):
    path = tmp_path / "train_history.json"
    logger = RewardCurveLogger(str(path), heldout_every=25, held_out_eval_fn=lambda **_: (0.55, 0.09))
    state = State(25)
    logger.on_log(None, state, None, logs={"reward": 0.5, "reward_std": 0.07})
    logger.on_step_end(None, state, None)

    data = json.loads(path.read_text())
    assert data["heldout_reward"] == [0.55]
    assert data["heldout_std"] == [0.09]


def test_reward_curve_logger_uses_atomic_replace():
    import inspect

    assert "os.replace" in inspect.getsource(RewardCurveLogger._write)
