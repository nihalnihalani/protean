"""Freeze Protean training tasks into a deterministic manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protean.manifest import freeze, meta_path_for  # noqa: E402
from protean.task_catalog import OPS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-split", type=int, default=12)
    parser.add_argument("--output", default=str(ROOT / "manifest_v1.jsonl"))
    parser.add_argument(
        "--no-sidecar",
        action="store_true",
        help="Do not write the <output>.meta.json integrity sidecar.",
    )
    args = parser.parse_args()

    sha = freeze(
        [op.name for op in OPS],
        n_per_split=args.n_per_split,
        path=args.output,
        write_sidecar=not args.no_sidecar,
    )
    print(f"wrote: {args.output}")
    print(f"sha256: {sha}")
    if not args.no_sidecar:
        print(f"sidecar: {meta_path_for(args.output)} (schema version + content hash + provenance)")
    print("set src/protean/tasks.py MANIFEST_SHA256 to this value to pin it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
