"""HUD task entrypoint plus a local manifest fallback."""

import os

from protean import env as _env_mod
from protean.manifest import build_manifest, load_frozen_manifest
from protean.task_catalog import OPS

env = _env_mod.env
kernel_opt = _env_mod.kernel_opt

for _op in OPS:
    globals()[_op.name] = getattr(_env_mod, _op.name)
for _task in _env_mod.HUD_TASKS:
    globals()[_task["id"]] = getattr(_env_mod, _task["id"])

MANIFEST_PATH = "manifest_v1.jsonl"
MANIFEST_SHA256 = "a5771e24f78b44d3a3406fb13e6342cdede382b8a634a56177a5db25ef666ed1"


def _load_tasks() -> list[dict]:
    allow_dynamic = os.environ.get("PROTEAN_ALLOW_DYNAMIC_MANIFEST") == "1"
    try:
        return load_frozen_manifest(
            MANIFEST_PATH,
            expected_sha256=MANIFEST_SHA256,
            allow_missing=allow_dynamic,
        )
    except RuntimeError:
        if allow_dynamic:
            return build_manifest(n_per_split=1)
        raise


TASKS = _load_tasks()
