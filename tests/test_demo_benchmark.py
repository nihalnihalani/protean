"""Tests for scripts/run_demo_benchmark.py (CPU-safe; never requires CUDA)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_demo_benchmark.py"

sys.path.insert(0, str(ROOT / "src"))

from protean.kernels import SEED_KERNELS
from protean.splits import HELD_OUT_SHAPES, TRAIN_SHAPES


def _cuda_present() -> bool:
    """Mirror the script's own notion of when a real benchmark is possible.

    The script returns exit code 2 only when at least one row carries the
    ``cuda_unavailable`` cap, which happens when torch/triton cannot be
    imported OR ``torch.cuda.is_available()`` is False. So a real GPU run is
    only expected when both imports succeed and a CUDA device is visible.
    """
    try:
        import torch  # type: ignore
        import triton  # noqa: F401  # type: ignore
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# When CUDA is genuinely present, exit code 2 (cuda_unavailable) would be a
# logic error, so the benchmark must actually succeed-or-fail (0/1). On a
# CPU-only box, code 2 is the only correct outcome. This keeps the test a real
# quality gate on GPU while staying CPU-safe.
_ALLOWED_EXIT_CODES = (0, 1) if _cuda_present() else (2,)


def _run(args: list[str], tmp_path: Path) -> tuple[int, list[dict], str]:
    json_out = tmp_path / "results.json"
    md_out = tmp_path / "results.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
            *args,
        ],
        capture_output=True,
        text=True,
    )
    rows = json.loads(json_out.read_text()) if json_out.exists() else []
    return proc.returncode, rows, md_out.read_text() if md_out.exists() else ""


def test_all_ops_covers_every_public_op(tmp_path):
    code, rows, md = _run(["--all-ops"], tmp_path)
    # CPU-only -> exit 2 (cuda_unavailable); a real GPU -> 0/1 (never 2).
    assert code in _ALLOWED_EXIT_CODES, (
        f"unexpected exit code {code}; cuda_present={_cuda_present()}"
    )

    ops_in_report = {row["op"] for row in rows}
    assert ops_in_report == set(SEED_KERNELS), (
        f"combined report missing ops: {set(SEED_KERNELS) - ops_in_report}"
    )

    # Every op is graded across both splits and all shapes.
    expected_per_op = len(TRAIN_SHAPES) + len(HELD_OUT_SHAPES)
    for op in SEED_KERNELS:
        op_rows = [r for r in rows if r["op"] == op]
        assert len(op_rows) == expected_per_op
        splits = {r["split"] for r in op_rows}
        assert splits == {"train", "held_out"}

    # Combined markdown includes the Op column header.
    assert "| Op |" in md


def test_all_ops_rows_have_required_fields(tmp_path):
    _, rows, _ = _run(["--all-ops"], tmp_path)
    assert rows
    for row in rows:
        for field in ("op", "split", "shape", "correct", "speedup", "reward", "caps"):
            assert field in row, f"row missing field {field!r}: {row}"
        assert isinstance(row["caps"], list)


def test_single_op_mode_unchanged(tmp_path):
    code, rows, md = _run(["--op", "rmsnorm"], tmp_path)
    assert code in _ALLOWED_EXIT_CODES, (
        f"unexpected exit code {code}; cuda_present={_cuda_present()}"
    )
    # Single-op mode grades exactly one op across both splits.
    assert {r["op"] for r in rows} == {"rmsnorm"}
    expected = len(TRAIN_SHAPES) + len(HELD_OUT_SHAPES)
    assert len(rows) == expected
    # Single-op markdown has no Op column.
    assert "| Op |" not in md
