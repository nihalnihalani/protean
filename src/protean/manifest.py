"""Small task manifest helpers for local, HUD, and GRPO runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from protean.sampler import sample_task
from protean.task_catalog import OPS


def build_manifest(n_per_split: int = 3, ops: list[str] | tuple[str, ...] | None = None) -> list[dict]:
    rows: list[dict] = []
    op_names = list(ops) if ops is not None else [op.name for op in OPS]
    for op_name in op_names:
        for split in ("train", "held_out"):
            for idx in range(n_per_split):
                rows.append(sample_task(op_name, idx, split))
    return rows


def write_manifest(path: str | Path, n_per_split: int = 3) -> None:
    p = Path(path)
    p.write_text("\n".join(json.dumps(row, sort_keys=True) for row in build_manifest(n_per_split)) + "\n")


def freeze(
    ops: list[str] | tuple[str, ...] | None = None,
    *,
    n_per_split: int = 12,
    path: str | Path = "manifest_v1.jsonl",
) -> str:
    """Write a deterministic manifest and return its SHA256."""

    p = Path(path)
    rows = build_manifest(n_per_split=n_per_split, ops=ops)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    os.replace(tmp, p)
    return sha256_file(p)


def sha256_file(path: str | Path) -> str:
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
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
