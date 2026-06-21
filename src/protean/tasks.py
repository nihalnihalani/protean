"""HUD taskset entrypoint plus the frozen reproducibility manifest.

This module carries two distinct, clearly-scoped concepts that share a single
source of truth (the ``(op, split)`` pairs declared by ``HUD_TASKS`` in
``protean.env``):

* ``tasks`` — the canonical module-level list of registered HUD ``Task`` objects
  (one per ``(op, split)``), mirroring ``verilog-template/tasks.py`` and
  ``coding-template/tasks.py``. This is what the file-based ``hud eval`` /
  ``hud sync`` path collects when it serves THIS module. ``env`` is re-exported
  so ``hud eval tasks.py`` can find the ``Environment`` on a local substrate.
  The intermediate Task objects are imported under underscore-prefixed aliases
  so HUD's taskset scanner (which skips ``_``-prefixed names and would otherwise
  also collect each bare attribute) does not double-count them as duplicate
  slugs.

* ``TASKS`` — the frozen reproducibility manifest: the per-``(op, split, seed)``
  expansion of those same pairs (3 ops x 2 splits x 12 seeds = 72 rows), loaded
  from the byte-pinned ``manifest_v1.jsonl``. This is the determinism backbone
  for local and GRPO grading. The HUD ``tasks`` list is the ``(op, split)`` view
  of exactly the same op/split universe that ``TASKS`` expands per seed; the
  ``tests/test_manifest.py`` suite asserts the two stay reconciled.

Run locally:  hud eval tasks.py claude --task-ids rmsnorm_held_out --group 1 -y
"""

import os

from protean.env import (
    elementwise_add_relu_held_out as _elementwise_add_relu_held_out,
)
from protean.env import (
    elementwise_add_relu_train as _elementwise_add_relu_train,
)
from protean.env import env  # noqa: F401  (env re-exported for `hud eval tasks.py`)
from protean.env import (
    rmsnorm_held_out as _rmsnorm_held_out,
)
from protean.env import (
    rmsnorm_train as _rmsnorm_train,
)
from protean.env import (
    softmax_rows_held_out as _softmax_rows_held_out,
)
from protean.env import (
    softmax_rows_train as _softmax_rows_train,
)
from protean.manifest import build_manifest, load_frozen_manifest

MANIFEST_PATH = "manifest_v1.jsonl"
MANIFEST_SHA256 = "a5771e24f78b44d3a3406fb13e6342cdede382b8a634a56177a5db25ef666ed1"


# Canonical taskset collected by `hud eval` / `hud sync tasks`. One concrete
# Task per (op, split). Intermediates above are underscore-prefixed so the
# scanner only counts each Task once (via this list), never twice.
#
# IMPORTANT (hud-present vs hud-absent): when the ``hud`` extra is installed
# (the deploy/serve image, and `hud eval`), each item below is a real HUD
# ``Task`` object — that is the only path HUD's taskset scanner ever runs on.
# When ``hud`` is ABSENT (CI's ``test`` extra), ``protean.env``'s else-branch
# assigns these names to plain ``dict`` stubs, so ``tasks`` then holds dicts,
# not ``Task`` objects. That is intentional and harmless (the scanner never
# runs without hud), but any hud-less caller iterating ``tasks`` must treat
# items as dicts, not Task objects (use ``t["id"]``/``t["op"]``, not ``t.slug``).
tasks = [
    _elementwise_add_relu_train,
    _elementwise_add_relu_held_out,
    _rmsnorm_train,
    _rmsnorm_held_out,
    _softmax_rows_train,
    _softmax_rows_held_out,
]


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
