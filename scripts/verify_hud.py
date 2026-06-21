#!/usr/bin/env python3
"""Validate the Protean HUD integration without requiring a GPU.

This is the pre-deploy smoke check for the ``hud serve`` contract. It proves
the env module imports, the ``@env.template`` decorators actually registered the
three Protean ops, and the registrations are discoverable from source the same
way ``hud deploy`` discovers them. None of these checks runs a kernel, so they
work on any machine (no CUDA required).

Steps (all GPU-free):
  1. Import ``protean.env`` and assert ``env`` is a live HUD Environment with the
     three expected templates registered (catches a silent zero-template deploy).
  2. Run ``hud task list --source src/protean/env.py`` and assert every expected
     task slug appears (mirrors how ``hud deploy``/the dashboard read the source).
  3. (Optional) If ``HUD_API_KEY`` is set, run the dashboard demo agent across all
     tasks. Skipped by default so CI/local verification needs no network or GPU.

Exit codes: 0 = all checks passed; 1 = a registration/listing check failed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_SOURCE = "src/protean/env.py"

# The three ops Protean serves. Each is exposed as a template plus a train/held_out
# task pair; both forms are asserted below.
EXPECTED_TEMPLATES = ("elementwise_add_relu", "rmsnorm", "softmax_rows")
EXPECTED_TASK_SLUGS = tuple(f"{op}_{split}" for op in EXPECTED_TEMPLATES for split in ("train", "held_out"))


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _check_import_and_templates() -> bool:
    """Import the env module and assert the templates registered at import time."""

    print("[protean] step 1/3: importing protean.env and checking template registration")
    snippet = (
        "import sys, json\n"
        "from protean.env import env\n"
        "if env is None:\n"
        "    print('FAIL: env is None — hud package not importable, zero templates', file=sys.stderr)\n"
        "    raise SystemExit(1)\n"
        "registered = sorted(getattr(env, 'tasks', {}) or {})\n"
        "expected = " + repr(sorted(EXPECTED_TEMPLATES)) + "\n"
        "print('  registered templates:', registered)\n"
        "missing = [t for t in expected if t not in registered]\n"
        "if missing:\n"
        "    print('FAIL: missing templates:', missing, file=sys.stderr)\n"
        "    raise SystemExit(1)\n"
        "print('  OK: all expected templates registered')\n"
    )
    result = subprocess.run([sys.executable, "-c", snippet], cwd=ROOT, env=_child_env())
    if result.returncode != 0:
        print("[protean] FAIL: template registration check failed", file=sys.stderr)
        return False
    return True


def _check_task_list_source() -> bool:
    """Run ``hud task list --source`` — the same source read path as ``hud deploy``."""

    print(f"[protean] step 2/3: hud task list --source {ENV_SOURCE}")
    cmd = [sys.executable, "-m", "hud", "task", "list", "--source", ENV_SOURCE]
    print("+", " ".join(cmd))
    try:
        result = subprocess.run(cmd, cwd=ROOT, env=_child_env(), capture_output=True, text=True)
    except FileNotFoundError:
        print("[protean] WARN: 'hud' CLI not available; skipping task-list check")
        return True

    sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        print("[protean] FAIL: 'hud task list --source' exited non-zero", file=sys.stderr)
        return False

    listed = result.stdout
    missing = [slug for slug in EXPECTED_TASK_SLUGS if slug not in listed]
    if missing:
        print(f"[protean] FAIL: tasks missing from source listing: {missing}", file=sys.stderr)
        return False
    print(f"  OK: all {len(EXPECTED_TASK_SLUGS)} expected task slugs present in source listing")
    return True


def _run_demo_agent() -> bool:
    """Optional dashboard run — only when HUD_API_KEY is set (needs network)."""

    if not os.environ.get("HUD_API_KEY"):
        print(
            "[protean] step 3/3: HUD_API_KEY not set — skipping dashboard demo agent "
            "(registration already verified offline)"
        )
        return True

    print("[protean] step 3/3: running HUD demo agent on all tasks")
    cmd = [sys.executable, "scripts/run_hud_demo_agent.py", "--source", ENV_SOURCE]
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, env=_child_env())
    if result.returncode != 0:
        print("[protean] FAIL: demo agent run failed", file=sys.stderr)
        return False
    return True


def main() -> int:
    checks = (
        _check_import_and_templates,
        _check_task_list_source,
        _run_demo_agent,
    )
    ok = True
    for check in checks:
        if not check():
            ok = False
            break
    if ok:
        print(
            f"[protean] HUD integration verified: 3 templates, {len(EXPECTED_TASK_SLUGS)} tasks register from source."
        )
        return 0
    print("[protean] HUD integration verification FAILED.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
