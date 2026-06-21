import json
from pathlib import Path

import pytest

from protean.manifest import (
    GENERATOR_COMMITTED,
    KNOWN_GENERATORS,
    SCHEMA_VERSION,
    build_manifest,
    compute_content_sha256,
    freeze,
    load_frozen_manifest,
    load_manifest_meta,
    meta_path_for,
    sha256_file,
)
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


# --- Integrity sidecar: schema version + content hash + provenance ---


def test_committed_manifest_has_valid_sidecar():
    meta = load_manifest_meta("manifest_v1.jsonl")
    assert meta is not None, "committed manifest must ship an integrity sidecar"
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["content_sha256"] == MANIFEST_SHA256
    assert meta["n_rows"] == len(TASKS) == 72


def test_freeze_writes_sidecar_with_schema_hash_and_provenance(tmp_path):
    path = tmp_path / "manifest.jsonl"
    sha = freeze(["elementwise_add_relu"], n_per_split=2, path=path)
    mp = meta_path_for(path)
    assert mp.exists()
    meta = json.loads(mp.read_text())
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["content_sha256"] == sha == sha256_file(path)
    assert meta["n_rows"] == 4
    assert meta["ops"] == ["elementwise_add_relu"]
    assert meta["n_per_split"] == 2
    prov = meta["provenance"]
    for key in ("created_at", "python_version", "platform", "generator"):
        assert key in prov


def test_freeze_load_roundtrip(tmp_path):
    path = tmp_path / "manifest.jsonl"
    freeze(["elementwise_add_relu", "rmsnorm"], n_per_split=3, path=path)
    rows = load_frozen_manifest(path)
    assert len(rows) == 12
    # Loading validates the sidecar; expected_sha256 also still works.
    rows2 = load_frozen_manifest(path, expected_sha256=sha256_file(path))
    assert rows == rows2


def test_content_hash_is_deterministic_and_matches_file(tmp_path):
    rows = build_manifest(n_per_split=4, ops=["elementwise_add_relu"])
    path = tmp_path / "manifest.jsonl"
    freeze(["elementwise_add_relu"], n_per_split=4, path=path)
    assert compute_content_sha256(rows) == sha256_file(path)


def test_load_detects_data_file_tampering_via_sidecar(tmp_path):
    path = tmp_path / "manifest.jsonl"
    freeze(["elementwise_add_relu"], n_per_split=2, path=path)
    # Tamper with the data file but leave the sidecar (recording the old hash).
    lines = path.read_text().splitlines()
    tampered = lines[0].replace('"seed": 0', '"seed": 99')
    path.write_text("\n".join([tampered] + lines[1:]) + "\n")
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        load_frozen_manifest(path)


def test_load_detects_schema_version_drift(tmp_path):
    path = tmp_path / "manifest.jsonl"
    freeze(["elementwise_add_relu"], n_per_split=1, path=path)
    mp = meta_path_for(path)
    meta = json.loads(mp.read_text())
    meta["schema_version"] = SCHEMA_VERSION + 1
    mp.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="schema version mismatch"):
        load_frozen_manifest(path)


def test_load_detects_row_count_drift(tmp_path):
    path = tmp_path / "manifest.jsonl"
    freeze(["elementwise_add_relu"], n_per_split=2, path=path)
    mp = meta_path_for(path)
    meta = json.loads(mp.read_text())
    meta["n_rows"] = 999  # claim more rows than the file holds
    mp.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="row count mismatch"):
        load_frozen_manifest(path)


def test_freeze_without_sidecar_still_loads(tmp_path):
    path = tmp_path / "manifest.jsonl"
    freeze(["elementwise_add_relu"], n_per_split=1, path=path, write_sidecar=False)
    assert not meta_path_for(path).exists()
    assert load_manifest_meta(path) is None
    # No sidecar means no extra validation; loading still works.
    rows = load_frozen_manifest(path)
    assert len(rows) == 2


def test_row_append_with_matching_n_rows_is_still_caught_by_content_hash(tmp_path):
    # An attacker appends a row AND updates n_rows to match. The content hash
    # (over the tampered file) must still fail, since they did not recompute it.
    path = tmp_path / "manifest.jsonl"
    freeze(["elementwise_add_relu"], n_per_split=2, path=path)
    mp = meta_path_for(path)
    meta = json.loads(mp.read_text())
    appended = path.read_text() + json.dumps({"injected": True}, sort_keys=True) + "\n"
    path.write_text(appended)
    meta["n_rows"] = meta["n_rows"] + 1  # forge n_rows to match the new file
    mp.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        load_frozen_manifest(path)


def test_crlf_mutation_triggers_hash_mismatch(tmp_path):
    # sha256_file hashes raw bytes (no CRLF normalization), so a CRLF<->LF
    # rewrite of an LF-canonical manifest is detected as drift, not masked.
    path = tmp_path / "manifest.jsonl"
    freeze(["elementwise_add_relu"], n_per_split=1, path=path)
    lf_sha = sha256_file(path)
    crlf = path.read_bytes().replace(b"\n", b"\r\n")
    assert crlf != path.read_bytes()
    path.write_bytes(crlf)
    assert sha256_file(path) != lf_sha
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        load_frozen_manifest(path)


def test_committed_sidecar_generator_is_known_and_bootstrap_is_explicit():
    meta = load_manifest_meta("manifest_v1.jsonl")
    assert meta is not None
    prov = meta["provenance"]
    # The generator must be a recognized value, not arbitrary free text.
    assert prov["generator"] in KNOWN_GENERATORS
    # The committed sidecar is the bootstrap case: null runtime provenance is
    # valid and explicitly flagged so auditors can distinguish it from a freeze.
    assert prov["generator"] == GENERATOR_COMMITTED
    assert prov["bootstrap"] is True
    for key in ("created_at", "python_version", "platform", "protean_version"):
        assert prov[key] is None


def test_freeze_sidecar_generator_is_known(tmp_path):
    path = tmp_path / "manifest.jsonl"
    freeze(["elementwise_add_relu"], n_per_split=1, path=path)
    meta = json.loads(meta_path_for(path).read_text())
    assert meta["provenance"]["generator"] in KNOWN_GENERATORS
