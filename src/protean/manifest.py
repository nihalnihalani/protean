"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# manifest.py — freeze to versioned JSONL; the loop consumes the FROZEN file, never re-samples
import os
import json
from .splits import assert_disjoint
from .sampler import sample_task

def _atomic_write(path, rows):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp_path, path)

def freeze(ops, n_per_op=12, path="manifest_v1.jsonl"):
    assert_disjoint()
    rows = []
    for op in ops:
        for i in range(n_per_op):
            rows.append(sample_task(op, i, "train"))
            rows.append(sample_task(op, i, "test"))
    _atomic_write(path, rows)

def load_frozen_manifest(path="manifest_v1.jsonl"):
    resolved_path = path
    if not os.path.isabs(path):
        # Try workspace path
        workspace_path = os.path.join(os.environ.get("WORKSPACE_ROOT", "/workdir"), path)
        if os.path.exists(workspace_path):
            resolved_path = workspace_path
        else:
            # Fallback to local package dir
            package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
            if os.path.exists(package_path):
                resolved_path = package_path

    if not os.path.exists(resolved_path):
        return []

    rows = []
    with open(resolved_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))
    return rows                    # write .tmp then os.replace (atomic, no half-read)
