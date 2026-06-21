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
from protean.splits import (
    HELD_OUT_SHAPES,
    STRUCTURAL_ANCHORS,
    TILING_BLOCK,
    TRAIN_SHAPES,
    is_offgrid,
)


def test_train_and_held_out_shapes_are_disjoint():
    assert set(TRAIN_SHAPES).isdisjoint(HELD_OUT_SHAPES)


def test_dev_held_out_shapes_are_off_the_tiling_grid():
    # G1 moat: a kernel hardcoding BLOCK=64/128/256/512 must NOT tile these evenly.
    for m in HELD_OUT_SHAPES:
        assert m % TILING_BLOCK != 0, f"{m} is on the {TILING_BLOCK}-tiling grid"
        for block in (64, 128, 256, 512):
            assert m % block != 0, f"{m} is a multiple of {block}"
        assert is_offgrid(m), f"{m} is not off-grid"


def test_structural_anchors_have_no_duplicates_and_include_dev_held_out():
    assert len(STRUCTURAL_ANCHORS) == len(set(STRUCTURAL_ANCHORS))
    assert set(HELD_OUT_SHAPES).issubset(set(STRUCTURAL_ANCHORS))


def test_structural_anchors_are_a_proper_superset_of_held_out():
    # Collapse / dead-export guard: STRUCTURAL_ANCHORS must carry coverage BEYOND the 3 dev held-out
    # shapes (otherwise the constant is a misleading no-op equal to HELD_OUT_SHAPES). Every anchor —
    # the dev shapes and the extra near-block-boundary anchors — must itself be off the tiling grid.
    assert len(STRUCTURAL_ANCHORS) > len(HELD_OUT_SHAPES), "anchors add no coverage beyond HELD_OUT_SHAPES"
    extra = set(STRUCTURAL_ANCHORS) - set(HELD_OUT_SHAPES)
    assert extra, "no structural anchors beyond the dev held-out shapes"
    for m in STRUCTURAL_ANCHORS:
        assert m % TILING_BLOCK != 0, f"{m} is on the {TILING_BLOCK}-tiling grid"
        assert is_offgrid(m), f"{m} is not off-grid"
        assert m not in TRAIN_SHAPES, f"{m} is on the train grid"


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
