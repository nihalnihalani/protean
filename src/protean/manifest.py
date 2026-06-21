"""Small task manifest helpers for local and HUD runs."""

from __future__ import annotations

import json
from pathlib import Path

from protean.sampler import sample_task
from protean.task_catalog import OPS


def build_manifest(n_per_split: int = 3) -> list[dict]:
    rows: list[dict] = []
    for op in OPS:
        for split in ("train", "held_out"):
            for idx in range(n_per_split):
                rows.append(sample_task(op.name, idx, split))
    return rows


def write_manifest(path: str | Path, n_per_split: int = 3) -> None:
    p = Path(path)
    p.write_text("\n".join(json.dumps(row, sort_keys=True) for row in build_manifest(n_per_split)) + "\n")


def load_frozen_manifest(path: str | Path = "manifest_v1.jsonl") -> list[dict]:
    p = Path(path)
    if not p.exists():
        return build_manifest(n_per_split=1)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
