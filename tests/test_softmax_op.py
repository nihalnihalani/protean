import importlib.util
import os
import sys

import pytest

sys.path.insert(0, "src")

from protean.task_catalog import OPS_BY_NAME

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _load_reference_module():
    from pathlib import Path

    path = Path(__file__).parent.parent / "src" / "protean" / "tasks" / "softmax_rows" / "donotaccess" / "reference.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("softmax_rows_ref", path)
    if spec and spec.loader:
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
        return ref
    return None


def test_softmax_op_registered():
    assert "softmax_rows" in OPS_BY_NAME
    op = OPS_BY_NAME["softmax_rows"]
    assert op.dtype == "float16"
    assert op.prompt_path == "src/protean/tasks/softmax_rows/prompt.md"


def test_softmax_files_exist():
    for fname in ("prompt.md", "donotaccess/grade.py", "donotaccess/reference.py"):
        path = os.path.join("src", "protean", "tasks", "softmax_rows", fname)
        assert os.path.exists(path), f"Missing: {path}"


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_softmax_reference_imports():
    ref = _load_reference_module()
    assert ref is not None
    assert callable(ref.eager_fn)
    assert callable(ref.make_inputs)


def test_manifest_includes_both_ops():
    from protean.manifest import load_frozen_manifest

    rows = load_frozen_manifest("manifest_v1.jsonl")
    op_names = {row["op"] for row in rows}
    assert "elementwise_add_relu" in op_names
    assert "softmax_rows" in op_names


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_softmax_eager_runs_on_cpu():
    """Sanity: the reference can at least construct the right shape on CPU.
    Full CUDA correctness is tested at Modal smoke-test time."""
    import torch

    ref = _load_reference_module()
    assert ref is not None

    x = torch.randn(64, 128, dtype=torch.float32)  # use fp32 + cpu for test
    out = ref.eager_fn(x)
    assert out.shape == x.shape
    # Softmax outputs sum to 1 along the last axis
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
