import json
from pathlib import Path

import pytest

from protean.manifest import freeze, load_frozen_manifest, sha256_file
from protean.splits import HELD_OUT_SHAPES, TRAIN_SHAPES
from protean.tasks import MANIFEST_SHA256, TASKS


def test_committed_manifest_is_pinned():
    assert Path("manifest_v1.jsonl").exists()
    assert sha256_file("manifest_v1.jsonl") == MANIFEST_SHA256
    assert len(TASKS) == 72


def test_freeze_is_deterministic(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    freeze(["elementwise_add_relu"], n_per_split=4, path=first)
    freeze(["elementwise_add_relu"], n_per_split=4, path=second)
    assert first.read_text() == second.read_text()


def test_freeze_preserves_split_shapes(tmp_path):
    path = tmp_path / "manifest.jsonl"
    freeze(["elementwise_add_relu"], n_per_split=8, path=path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 16
    for row in rows:
        if row["split"] == "train":
            assert row["shape"] in TRAIN_SHAPES
        else:
            assert row["shape"] in HELD_OUT_SHAPES


def test_load_manifest_hash_mismatch_fails(tmp_path):
    path = tmp_path / "manifest.jsonl"
    freeze(["elementwise_add_relu"], n_per_split=1, path=path)
    with pytest.raises(RuntimeError, match="Manifest hash mismatch"):
        load_frozen_manifest(path, expected_sha256="bad")


def test_load_missing_manifest_can_be_allowed(tmp_path):
    assert load_frozen_manifest(tmp_path / "missing.jsonl", allow_missing=True) == []
