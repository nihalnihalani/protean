"""Freeze the task manifest to a versioned JSONL for reproducible training.

Run this script whenever splits.py, the op catalog, or the curriculum changes.
The resulting manifest_v1.jsonl should be committed to the repo and the trainer
loads it at runtime — NEVER re-samples dynamically in production.

Usage:
    python scripts/freeze_manifest.py [--n-per-op 12] [--output manifest_v1.jsonl]
"""
import argparse
import hashlib
import json
import os
import sys

# Setup paths
local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(local_dir, "src"))

from protean.manifest import freeze
from protean.task_catalog import OPS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-op", type=int, default=12,
                        help="Tasks per op per split. Default 12 → 24 rows per op (train+test).")
    parser.add_argument("--output", type=str, default=os.path.join(local_dir, "manifest_v1.jsonl"),
                        help="Output path. Convention: manifest_v<N>.jsonl, bump N on schema change.")
    args = parser.parse_args()

    op_names = [op.op_name for op in OPS]
    print(f"[freeze] Ops: {op_names}")
    print(f"[freeze] Tasks per op per split: {args.n_per_op}")
    print(f"[freeze] Total rows: {len(op_names) * args.n_per_op * 2} (train + test)")

    freeze(op_names, n_per_op=args.n_per_op, path=args.output)

    # Hash the result so future runs can verify they're using the same manifest
    with open(args.output, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    print(f"[freeze] Wrote {args.output} ({os.path.getsize(args.output)} bytes)")
    print(f"[freeze] sha256: {sha}")
    print(f"[freeze] Commit this hash to docs/ if you want to make it part of the audit trail.")

    # Sanity check: confirm the JSONL is well-formed
    row_count = 0
    train_count = 0
    test_count = 0
    with open(args.output, "r") as f:
        for line in f:
            row = json.loads(line)
            row_count += 1
            if row["split"] == "train":
                train_count += 1
            elif row["split"] == "test":
                test_count += 1

    print(f"[freeze] Verified: {row_count} rows, {train_count} train, {test_count} test")
    assert train_count == test_count, "Train/test row counts must be equal for unbiased eval"

    print()
    print(f"[freeze] To pin this manifest in src/protean/tasks.py, add this constant:")
    print(f"    MANIFEST_SHA256 = \"{sha}\"")
    print(f"[freeze] (Currently the trainer warns on mismatch; pin to make it a hard error.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
