import importlib.util

from protean.grader import grade_source
from protean.kernels import (
    BAD_SHAPE_TRITON,
    HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU,
    HAND_OPTIMIZED_RMSNORM,
    NO_LAUNCH,
    PYTORCH_PASSTHROUGH,
    PYTORCH_RMSNORM_PASSTHROUGH,
)
from protean.splits import HELD_OUT_SHAPES, TRAIN_SHAPES


def test_train_and_held_out_shapes_are_disjoint():
    assert set(TRAIN_SHAPES).isdisjoint(HELD_OUT_SHAPES)


def test_pytorch_passthrough_rejected_statically():
    grade = grade_source(PYTORCH_PASSTHROUGH)
    assert grade["reward"] == 0.0
    assert any(cap.startswith("ast_ban") for cap in grade["caps"])


def test_pytorch_rmsnorm_passthrough_rejected_statically():
    grade = grade_source(PYTORCH_RMSNORM_PASSTHROUGH, op="rmsnorm")
    assert grade["reward"] == 0.0
    assert any(cap.startswith("ast_ban") for cap in grade["caps"])


def test_no_launch_rejected_statically():
    grade = grade_source(NO_LAUNCH)
    assert grade["reward"] == 0.0
    assert "no_triton_jit" in grade["caps"]


def test_known_good_kernel_passes_when_cuda_available():
    if importlib.util.find_spec("torch") is None:
        grade = grade_source(HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU)
        assert "cuda_unavailable" in grade["caps"]
        return
    import torch

    if not torch.cuda.is_available():
        grade = grade_source(HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU)
        assert "cuda_unavailable" in grade["caps"]
        return
    grade = grade_source(HAND_OPTIMIZED_ELEMENTWISE_ADD_RELU, reps=5, warmup=2)
    assert grade["correct"] is True
    assert grade["reward"] > 0


def test_known_good_rmsnorm_passes_when_cuda_available():
    if importlib.util.find_spec("torch") is None:
        grade = grade_source(HAND_OPTIMIZED_RMSNORM, op="rmsnorm")
        assert "cuda_unavailable" in grade["caps"]
        return
    import torch

    if not torch.cuda.is_available():
        grade = grade_source(HAND_OPTIMIZED_RMSNORM, op="rmsnorm")
        assert "cuda_unavailable" in grade["caps"]
        return
    grade = grade_source(HAND_OPTIMIZED_RMSNORM, op="rmsnorm", reps=5, warmup=2)
    assert grade["correct"] is True
    assert grade["reward"] > 0


def test_bad_shape_rejected_when_cuda_available():
    grade = grade_source(BAD_SHAPE_TRITON, reps=5, warmup=2)
    if importlib.util.find_spec("torch") is None:
        assert "cuda_unavailable" in grade["caps"]
        return
    import torch

    if not torch.cuda.is_available():
        assert "cuda_unavailable" in grade["caps"]
        return
    assert grade["reward"] == 0.0
    assert "shape_mismatch" in grade["caps"]
