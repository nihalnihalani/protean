"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# manifest.py — freeze to versioned JSONL; the loop consumes the FROZEN file, never re-samples
import json, os
from protean.splits import _assert_split_disjoint   # FIX (devil's-advocate R2): missing imports -> NameError
from protean.sampler import sample_task


def _atomic_write(path, rows):
    """Write .tmp then os.replace (atomic; no half-read by a concurrent rollout worker)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def load_frozen_manifest(path="manifest_v1.jsonl"):
    """Consumed by tasks.py; the loop reads this frozen file and never re-samples shapes."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def freeze(ops, n_per_op=12, path="manifest_v1.jsonl"):
    _assert_split_disjoint()
    rows = []
    for op in ops:
        for i in range(n_per_op):
            rows.append(sample_task(op, i, "train"))
            rows.append(sample_task(op, i, "test"))
    _atomic_write(path, rows)                    # write .tmp then os.replace (atomic, no half-read)
