"""Taskset that `hud eval src/protean/tasks.py <model>` serves and the trainer's
Taskset.run() consumes. Enumerates op × split → Task. See IMPLEMENTATION_PLAN §4.2.
"""
from protean.env import env, kernel_opt  # noqa: F401  (re-export so `hud eval` finds the template)
from protean.task_catalog import OPS
from protean.manifest import load_frozen_manifest

# The trainer/eval consume the FROZEN manifest (never re-sample shapes at runtime — that breaks the split invariant).
# Build it once with: python -m protean.manifest --freeze
TASKS = load_frozen_manifest()  # list of {op_name, split, seed, slug, columns}

# TODO: if no frozen manifest exists yet, fall back to enumerating train+held_out per op for smoke tests:
#   TASKS = [dict(op_name=o.op_name, split=s, seed=i) for o in OPS for s in ("train","held_out") for i in range(N)]
