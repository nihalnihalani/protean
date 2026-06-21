"""Tests for the manifest freeze/load pipeline (Step 11).

Validates:
  - freeze() produces valid JSONL with correct row counts
  - Train shapes ∈ TRAIN_M, test shapes ∈ TEST_M (split invariant)
  - Deterministic: same inputs produce identical output
  - load_frozen_manifest returns [] for missing files
  - Manifest rows include M, N, op, split, dtype fields
"""
import os
import json
import tempfile
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from protean.manifest import freeze, load_frozen_manifest
from protean.splits import TRAIN_M, TEST_M


def test_freeze_produces_valid_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_manifest.jsonl")
        freeze(["elementwise_add_relu"], n_per_op=4, path=path)

        assert os.path.exists(path)

        rows = []
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))

        # 4 train + 4 test = 8 rows for one op
        assert len(rows) == 8
        assert sum(1 for r in rows if r["split"] == "train") == 4
        assert sum(1 for r in rows if r["split"] == "test") == 4


def test_freeze_respects_split_invariant():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_manifest.jsonl")
        freeze(["elementwise_add_relu"], n_per_op=8, path=path)

        rows = []
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))

        for row in rows:
            if row["split"] == "train":
                assert row["M"] in TRAIN_M, f"Train M={row['M']} not in TRAIN_M={TRAIN_M}"
                assert row["N"] in TRAIN_M
            elif row["split"] == "test":
                assert row["M"] in TEST_M, f"Test M={row['M']} not in TEST_M={TEST_M}"
                assert row["N"] in TEST_M


def test_freeze_is_deterministic():
    """Same op + idx + split must produce same shape across freeze calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = os.path.join(tmpdir, "m1.jsonl")
        path2 = os.path.join(tmpdir, "m2.jsonl")
        freeze(["elementwise_add_relu"], n_per_op=4, path=path1)
        freeze(["elementwise_add_relu"], n_per_op=4, path=path2)

        with open(path1) as f1, open(path2) as f2:
            r1 = [json.loads(line) for line in f1]
            r2 = [json.loads(line) for line in f2]

        assert r1 == r2, "Freeze must be deterministic"


def test_load_returns_empty_when_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "does_not_exist.jsonl")
        rows = load_frozen_manifest(path)
        assert rows == []


def test_freeze_roundtrip():
    """Freeze then load must return the same rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "roundtrip.jsonl")
        freeze(["elementwise_add_relu"], n_per_op=4, path=path)
        rows = load_frozen_manifest(path)
        assert len(rows) == 8
        # Verify each row has required fields
        for row in rows:
            assert "op" in row
            assert "split" in row
            assert "M" in row
            assert "N" in row
            assert "dtype" in row


def test_freeze_multi_op():
    """Freeze with multiple ops produces correct row counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "multi.jsonl")
        # Even though we only have one op in the catalog, test the iteration
        freeze(["elementwise_add_relu"], n_per_op=6, path=path)
        rows = load_frozen_manifest(path)
        assert len(rows) == 12  # 6 train + 6 test
        train_rows = [r for r in rows if r["split"] == "train"]
        test_rows = [r for r in rows if r["split"] == "test"]
        assert len(train_rows) == len(test_rows) == 6


def test_manifest_rows_have_shapes():
    """Each manifest row must include M and N dimensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "shapes.jsonl")
        freeze(["elementwise_add_relu"], n_per_op=4, path=path)
        rows = load_frozen_manifest(path)
        for row in rows:
            assert isinstance(row["M"], int), f"M should be int, got {type(row['M'])}"
            assert isinstance(row["N"], int), f"N should be int, got {type(row['N'])}"
            assert row["M"] > 0
            assert row["N"] > 0
