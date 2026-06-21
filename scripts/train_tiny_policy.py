#!/usr/bin/env python3
"""Train the v1 small policy head from optimizer verifier traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from protean.model.tiny_policy import DEFAULT_HIDDEN_DIM, train_tiny_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="runs/protean-overnight/trials.jsonl")
    parser.add_argument("--out", default="runs/protean-overnight/tiny_policy.json")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    metrics = train_tiny_policy(
        ROOT / args.trace,
        ROOT / args.out,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
