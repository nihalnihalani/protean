#!/usr/bin/env python3
"""Run the HUD verification path for all Protean public tasks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    if not os.environ.get("HUD_API_KEY"):
        print("HUD_API_KEY is required for HUD dashboard job creation.", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    print("[protean] verifying HUD task metadata")
    subprocess.run([sys.executable, "-m", "protean.env"], cwd=ROOT, env=env, check=True)

    print("[protean] running HUD demo agent on all tasks")
    _run([sys.executable, "scripts/run_hud_demo_agent.py", "--source", "src/protean/env.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
