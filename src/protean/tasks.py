"""Taskset that `hud eval src/protean/tasks.py <model>` serves and the trainer's
Taskset.run() consumes. Enumerates op × split → Task. See IMPLEMENTATION_PLAN §4.2.
"""
from .env import env, kernel_opt  # noqa: F401
from .task_catalog import OPS
from .manifest import load_frozen_manifest

frozen_rows = load_frozen_manifest()

# Fallback: if no frozen manifest exists yet, generate dynamically
if not frozen_rows:
    frozen_rows = []
    for o in OPS:
        for split in ("train", "held_out"):
            for seed in range(3):
                frozen_rows.append({
                    "op_name": o.op_name,
                    "split": split,
                    "seed": seed,
                })

tasks = []
for row in frozen_rows:
    op_name = row.get("op_name") or row.get("op")
    split = row.get("split")
    seed = row.get("seed", 0)
    
    t = kernel_opt(op_name=op_name, split=split, seed=seed)
    t.slug = f"{op_name}-{split}-s{seed}"
    t.columns = {"op": op_name, "split": split, "seed": str(seed)}
    tasks.append(t)

TASKS = tasks
