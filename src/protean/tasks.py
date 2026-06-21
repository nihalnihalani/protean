"""HUD task entrypoint plus a local manifest fallback."""

from protean.env import env, kernel_opt  # noqa: F401
from protean.manifest import load_frozen_manifest

TASKS = load_frozen_manifest()
