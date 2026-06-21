"""HUD task entrypoint plus a local manifest fallback."""

from protean.env import (  # noqa: F401
    elementwise_add_relu_held_out,
    elementwise_add_relu_train,
    env,
    kernel_opt,
    rmsnorm_held_out,
    rmsnorm_train,
)
from protean.manifest import load_frozen_manifest

TASKS = load_frozen_manifest()
