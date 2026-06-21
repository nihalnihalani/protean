from train.callbacks import CostAbortCallback


class State:
    def __init__(self, step: int, logs=None):
        self.global_step = step
        self.log_history = logs or []


class Control:
    should_training_stop = False


def test_step_review_fails_on_low_mean():
    callback = CostAbortCallback(review_step=10, min_reward_at_review=0.1)
    callback.reward_history = [0.01] * 30
    control = Control()
    callback.on_step_end(None, State(10), control)
    assert control.should_training_stop is True


def test_step_review_passes_with_variance():
    callback = CostAbortCallback(review_step=10, min_reward_at_review=0.1, min_reward_std_at_review=0.02)
    callback.reward_history = [0.2, 0.4, 0.6, 0.3, 0.5] * 6
    control = Control()
    callback.on_step_end(None, State(10), control)
    assert control.should_training_stop is False


def test_flatline_window_is_30():
    callback = CostAbortCallback()
    callback.reward_history = [0.5] * 29
    assert callback._reward_flat(window=30) is False
    callback.reward_history.append(0.5)
    assert callback._reward_flat(window=30) is True
