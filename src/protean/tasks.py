"""HUD task entrypoint plus a local manifest fallback."""

import os

from protean.env import (  # noqa: F401
    elementwise_add_relu_held_out,
    elementwise_add_relu_train,
    env,
    kernel_opt,
    rmsnorm_held_out,
    rmsnorm_train,
    softmax_rows_held_out,
    softmax_rows_train,
)
from protean.manifest import build_manifest, load_frozen_manifest

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
