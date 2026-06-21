from pathlib import Path

from protean.bench_core import eager_fn_for_op
from protean.kernels import SEED_KERNELS
from protean.task_catalog import OPS, OPS_BY_NAME


COMPANY_OPS = {
    "matmul_tile",
    "attention_softmax",
    "layernorm",
    "fused_mlp",
    "quantize_dequant",
    "moe_routing",
    "embedding_lookup",
    "sum_reduction",
    "prefix_scan",
}


def test_company_ops_registered():
    assert COMPANY_OPS <= set(OPS_BY_NAME)


def test_every_registered_op_has_public_task_files():
    root = Path("src/protean/tasks")
    for op in OPS:
        task_dir = root / op.name
        assert (task_dir / "prompt.md").exists(), op.name
        assert (task_dir / "donotaccess" / "grade.py").exists(), op.name
        assert (task_dir / "donotaccess" / "reference.py").exists(), op.name


def test_every_registered_op_has_seed_and_eager_reference():
    for op in OPS:
        assert op.name in SEED_KERNELS
        assert "def solution" in SEED_KERNELS[op.name]
        assert callable(eager_fn_for_op(op.name))

