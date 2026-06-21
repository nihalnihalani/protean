"""Taskset that `hud eval src/protean/tasks.py <model>` serves and the trainer's
Taskset.run() consumes. Enumerates op × split → Task. See IMPLEMENTATION_PLAN §4.2.

PRODUCTION: requires manifest_v<N>.jsonl to exist. Dynamic generation is only used
when PROTEAN_ALLOW_DYNAMIC_MANIFEST=1 (local dev escape hatch). This prevents
silent "lost reproducibility" failures where someone deletes the manifest and
training continues with random shapes.
"""
import os
import hashlib
from .env import env, kernel_opt  # noqa: F401
from .task_catalog import OPS
from .manifest import load_frozen_manifest

_MANIFEST_VERSION = "v1"
_MANIFEST_PATH = f"manifest_{_MANIFEST_VERSION}.jsonl"

# Pin the expected manifest hash. Update via: python scripts/freeze_manifest.py
# Set to the sha256 printed by the freeze script to enforce integrity checking.
# While None, the check is a no-op — set it once you have a canonical frozen manifest.
MANIFEST_SHA256 = "86290d88a65d7381fa5f93aa991cf1c26e679d19ba360fe0152c9f9529f1d66b"

frozen_rows = load_frozen_manifest(_MANIFEST_PATH)

if not frozen_rows:
    if os.environ.get("PROTEAN_ALLOW_DYNAMIC_MANIFEST") == "1":
        print(f"[protean] WARNING: {_MANIFEST_PATH} not found, using dynamic generation.")
        print(f"[protean] WARNING: This is NOT reproducible. Use only for local dev/testing.")
        print(f"[protean] WARNING: Production must run: python scripts/freeze_manifest.py")
        # Fallback: dynamic generation, same as before
        frozen_rows = []
        for o in OPS:
            for split in ("train", "test"):
                for seed in range(3):
                    frozen_rows.append({
                        "op": o.op_name,
                        "split": split,
                        "seed": seed,
                    })
    else:
        raise RuntimeError(
            f"Protean manifest {_MANIFEST_PATH} not found. "
            f"Production training requires a frozen manifest for reproducibility. "
            f"Generate one with: python scripts/freeze_manifest.py "
            f"Or set PROTEAN_ALLOW_DYNAMIC_MANIFEST=1 to bypass (local dev only)."
        )

# Manifest hash integrity check (step 11)
if frozen_rows and MANIFEST_SHA256 is not None:
    # Find the actual manifest file to hash
    _resolved = _MANIFEST_PATH
    if not os.path.isabs(_MANIFEST_PATH):
        _ws = os.path.join(os.environ.get("WORKSPACE_ROOT", "/workdir"), _MANIFEST_PATH)
        if os.path.exists(_ws):
            _resolved = _ws
        else:
            _pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", _MANIFEST_PATH)
            if os.path.exists(_pkg):
                _resolved = _pkg
    if os.path.exists(_resolved):
        with open(_resolved, "rb") as _f:
            _actual_sha = hashlib.sha256(_f.read()).hexdigest()
        if _actual_sha != MANIFEST_SHA256:
            raise RuntimeError(
                f"Manifest hash mismatch! "
                f"Expected {MANIFEST_SHA256}, got {_actual_sha}. "
                f"The manifest was modified after being pinned. Re-run scripts/freeze_manifest.py "
                f"and update MANIFEST_SHA256 in tasks.py if the change is intentional."
            )

tasks = []
for row in frozen_rows:
    op_name = row.get("op_name") or row.get("op")
    split = row.get("split")
    seed = row.get("seed", 0)
    M = row.get("M")
    N = row.get("N")

    t = kernel_opt(op_name=op_name, split=split, seed=seed)
    t.slug = f"{op_name}-{split}-s{seed}"
    # Include M and N in columns so build_dataset() can read them directly
    # from the manifest rather than re-sampling (step 11 reproducibility).
    t.columns = {"op": op_name, "split": split, "seed": str(seed)}
    if M is not None:
        t.columns["M"] = str(M)
    if N is not None:
        t.columns["N"] = str(N)
    tasks.append(t)

TASKS = tasks
