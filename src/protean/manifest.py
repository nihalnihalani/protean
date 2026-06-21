"""Small task manifest helpers for local, HUD, and GRPO runs.

Frozen manifests are the reproducibility backbone of Protean evaluation: the
exact set of (op, shape, split, seed) rows must be identical across machines and
across time, or any cross-run comparison is meaningless. To make drift detectable
rather than silent, every frozen manifest is paired with a sidecar metadata file
(``<manifest>.meta.json``) carrying a schema version, a content hash of the data
file, the row count, and provenance. Loaders validate the hash and fail loudly on
any mismatch.

The data file itself (``manifest_v1.jsonl``) is a plain JSONL stream of task rows
and is kept byte-stable so existing pinned SHAs remain valid; all integrity
metadata lives in the sidecar, so callers that only want rows are unaffected.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from protean.sampler import sample_task
from protean.task_catalog import OPS

# Bump when the manifest row schema or sidecar metadata schema changes in a way
# that should invalidate previously frozen manifests.
SCHEMA_VERSION = 1

# Suffix appended to a manifest path to locate/write its integrity sidecar.
META_SUFFIX = ".meta.json"

# Stable identifier written by the canonical freeze() pipeline into the sidecar
# provenance. Used to distinguish a freshly frozen manifest from the bootstrap
# committed sidecar.
GENERATOR_FREEZE = "protean.manifest.freeze"

# Identifier for the bootstrap sidecar that was generated for the pre-existing
# committed manifest_v1.jsonl WITHOUT re-running freeze() (the committed data
# file is byte-pinned to MANIFEST_SHA256). Its provenance fields are
# legitimately null because no live freeze run produced them; only the
# integrity fields (schema_version, content_sha256, n_rows) carry guarantees.
GENERATOR_COMMITTED = "committed (pinned manifest_v1.jsonl)"

# Enumerated set of recognized generators. An auditor can assert the sidecar's
# generator is one of these; anything else is an unrecognized/forged sidecar.
KNOWN_GENERATORS = frozenset({GENERATOR_FREEZE, GENERATOR_COMMITTED})


def build_manifest(n_per_split: int = 3, ops: list[str] | tuple[str, ...] | None = None) -> list[dict]:
    rows: list[dict] = []
    op_names = list(ops) if ops is not None else [op.name for op in OPS]
    for op_name in op_names:
        for split in ("train", "held_out"):
            for idx in range(n_per_split):
                rows.append(sample_task(op_name, idx, split))
    return rows


def canonical_manifest_bytes(rows: list[dict]) -> bytes:
    """Deterministic byte serialization of manifest rows (LF-terminated JSONL).

    This is the single source of truth for what gets written to disk and what
    gets hashed, so the on-disk file and the recorded content hash can never
    disagree.
    """

    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    return text.encode("utf-8")


def compute_content_sha256(rows: list[dict]) -> str:
    """SHA256 over the canonical serialization of ``rows``.

    Matches ``sha256_file`` of a file written from the same rows.
    """

    return hashlib.sha256(canonical_manifest_bytes(rows)).hexdigest()


def write_manifest(path: str | Path, n_per_split: int = 3) -> None:
    p = Path(path)
    p.write_bytes(canonical_manifest_bytes(build_manifest(n_per_split)))


def meta_path_for(path: str | Path) -> Path:
    """Return the sidecar metadata path for a manifest data file path."""

    p = Path(path)
    return p.with_name(p.name + META_SUFFIX)


def build_meta(
    rows: list[dict],
    *,
    ops: list[str] | tuple[str, ...] | None = None,
    n_per_split: int,
    include_runtime_provenance: bool = True,
) -> dict:
    """Build the integrity + provenance envelope for a set of manifest rows.

    Deterministic integrity fields (``schema_version``, ``content_sha256``,
    ``n_rows``) are always present and never depend on wall-clock or host state,
    so they are safe to compare across runs. Non-deterministic provenance
    (timestamp, python version, platform, package version) is grouped under
    ``provenance`` and is for human/audit use only; it is never part of the
    integrity check.
    """

    op_names = list(ops) if ops is not None else [op.name for op in OPS]
    meta: dict = {
        "schema_version": SCHEMA_VERSION,
        "content_sha256": compute_content_sha256(rows),
        "n_rows": len(rows),
        "ops": sorted(op_names),
        "n_per_split": n_per_split,
    }
    if include_runtime_provenance:
        meta["provenance"] = _runtime_provenance()
    return meta


def _runtime_provenance() -> dict:
    try:
        version = importlib.metadata.version("protean")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "protean_version": version,
        "generator": GENERATOR_FREEZE,
    }


def write_meta(path: str | Path, meta: dict) -> Path:
    """Atomically write a sidecar metadata file next to ``path``; return its path."""

    mp = meta_path_for(path)
    tmp = mp.with_suffix(mp.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
    os.replace(tmp, mp)
    return mp


def freeze(
    ops: list[str] | tuple[str, ...] | None = None,
    *,
    n_per_split: int = 12,
    path: str | Path = "manifest_v1.jsonl",
    write_sidecar: bool = True,
) -> str:
    """Write a deterministic manifest (+ integrity sidecar) and return its SHA256.

    The data file is written atomically and byte-deterministically. When
    ``write_sidecar`` is true (default) a ``<path>.meta.json`` integrity file is
    also written carrying the schema version, content hash, row count, and
    provenance.
    """

    p = Path(path)
    rows = build_manifest(n_per_split=n_per_split, ops=ops)
    data = canonical_manifest_bytes(rows)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    sha = sha256_file(p)
    if write_sidecar:
        meta = build_meta(rows, ops=ops, n_per_split=n_per_split)
        write_meta(p, meta)
    return sha


def sha256_file(path: str | Path) -> str:
    """SHA256 over the raw bytes of ``path``.

    No line-ending normalization is performed: ``freeze`` always writes
    ``canonical_manifest_bytes`` (LF-terminated, atomically) so a committed
    manifest never carries CRLFs. Hashing raw bytes means *any* mutation of the
    file -- including a CRLF<->LF rewrite (e.g. git autocrlf on Windows) -- is
    detected as drift rather than silently masked. Matches
    ``compute_content_sha256`` for files written from the same rows.
    """

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_manifest_path(path: str | Path = "manifest_v1.jsonl") -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        Path.cwd() / p,
        Path(__file__).resolve().parents[2] / p,
        Path(os.environ.get("WORKSPACE_ROOT", "")) / p if os.environ.get("WORKSPACE_ROOT") else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return p


def load_manifest_meta(path: str | Path = "manifest_v1.jsonl") -> dict | None:
    """Load the sidecar integrity metadata for a manifest, or None if absent.

    ``path`` is the manifest *data* file path; the sidecar is resolved as
    ``<resolved path>.meta.json``.
    """

    p = resolve_manifest_path(path)
    mp = meta_path_for(p)
    if not mp.exists():
        return None
    return json.loads(mp.read_text())


def _validate_meta(p: Path, meta: dict, rows: list[dict] | None = None) -> None:
    """Fail loudly if the sidecar metadata does not match the data file on disk.

    This is the single integrity gate. It checks schema version, content hash,
    and -- when ``rows`` is provided (the parsed data file) -- the row count, so
    all three integrity invariants are enforced together rather than scattered
    across the caller. An attacker who rewrites the data file must defeat the
    content hash regardless of how the row count is patched.
    """

    schema = meta.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise RuntimeError(
            f"Manifest schema version mismatch for {p}: sidecar declares "
            f"{schema!r}, loader supports {SCHEMA_VERSION!r}. Re-freeze the manifest."
        )
    declared_sha = meta.get("content_sha256")
    if declared_sha is not None:
        actual = sha256_file(p)
        if actual != declared_sha:
            raise RuntimeError(
                f"Manifest content hash mismatch for {p}: sidecar records "
                f"{declared_sha}, file is {actual}. The manifest was modified "
                "after freezing (tampering or accidental edit)."
            )
    if rows is not None:
        declared_rows = meta.get("n_rows")
        if declared_rows is not None and declared_rows != len(rows):
            raise RuntimeError(
                f"Manifest row count mismatch for {p}: sidecar records {declared_rows} rows, file has {len(rows)}."
            )


def load_frozen_manifest(
    path: str | Path = "manifest_v1.jsonl",
    *,
    expected_sha256: str | None = None,
    allow_missing: bool = False,
) -> list[dict]:
    p = resolve_manifest_path(path)
    if not p.exists():
        if allow_missing:
            return []
        raise RuntimeError(
            f"Protean manifest {path} not found. Generate it with "
            "`python scripts/freeze_manifest.py` or set PROTEAN_ALLOW_DYNAMIC_MANIFEST=1."
        )
    if expected_sha256 is not None:
        actual = sha256_file(p)
        if actual != expected_sha256:
            raise RuntimeError(f"Manifest hash mismatch: expected {expected_sha256}, got {actual}")
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    # Defense in depth: if an integrity sidecar exists, validate it too. This
    # catches drift even when the caller did not pass an expected_sha256. A
    # single gate checks schema, content hash, and row count together.
    mp = meta_path_for(p)
    if mp.exists():
        meta = json.loads(mp.read_text())
        _validate_meta(p, meta, rows=rows)
    return rows
