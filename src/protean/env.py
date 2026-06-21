"""HUD public wrapper around the direct Protean verifier."""

# NOTE: this file deliberately omits ``from __future__ import annotations``. Do NOT add it
# back: under it, an ``@env.template`` parameter typed with ``int | None`` (or any
# Literal/Optional/alias/Pydantic model) crashes at deploy/start because the HUD server
# runs ``TypeAdapter`` on a string forward-ref -> PydanticUserError, surfaced as JSON-RPC
# ``-32000``. Without it, annotations resolve to real objects and any param type works.
# Python 3.12 supports ``int | None``/``str | None`` natively (PEP 604), so removal is
# runtime-safe. Mirrors the verilog-template guidance. Leave the future-import out.

import json
import os
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from protean.grader import ProteanValidationError, grade_source, to_eval_result
from protean.splits import Split, shapes_for_split
from protean.task_catalog import get_op

if TYPE_CHECKING:
    from hud import Environment as _Environment

    Environment: type[_Environment] | None
try:
    from hud import Environment
except Exception:  # pragma: no cover - local tests should not require HUD.
    Environment = None


HUD_TASKS = (
    {"id": "elementwise_add_relu_train", "op": "elementwise_add_relu", "split": "train"},
    {"id": "elementwise_add_relu_held_out", "op": "elementwise_add_relu", "split": "held_out"},
    {"id": "rmsnorm_train", "op": "rmsnorm", "split": "train"},
    {"id": "rmsnorm_held_out", "op": "rmsnorm", "split": "held_out"},
    {"id": "softmax_rows_train", "op": "softmax_rows", "split": "train"},
    {"id": "softmax_rows_held_out", "op": "softmax_rows", "split": "held_out"},
)


def _as_split(split: str) -> Split:
    """Validate a public ``str`` split at the boundary and narrow it to ``Split``.

    The public wrappers accept ``split: str`` (no public-API drift), but the
    downstream consumers (``shapes_for_split``/``grade_source``) require the
    ``Split`` literal. Guard here so a bad value raises the typed
    ``ProteanValidationError`` from the grader's error hierarchy rather than a
    bare ``ValueError`` deep inside ``shapes_for_split``. After the membership
    check mypy narrows ``split`` to ``Split`` on the return path, so no ``cast``
    is needed.
    """

    valid: tuple[Split, ...] = get_args(Split)
    if split not in valid:
        raise ProteanValidationError(f"split {split!r} must be 'train' or 'held_out'")
    return split


def configure_triton_cache_dir() -> str:
    """Use Docker's warmed cache when available, otherwise a user cache."""

    current = os.environ.get("TRITON_CACHE_DIR")
    candidates = [Path(current)] if current else []
    candidates.append(Path("/triton-cache"))
    candidates.append(Path.home() / ".cache" / "protean-triton")

    for path in candidates:
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".write-test"
            test_file.write_text("ok")
            test_file.unlink()
            os.environ["TRITON_CACHE_DIR"] = str(path)
            return str(path)
        except Exception:
            continue
    raise RuntimeError("no writable Triton cache directory found")


# Configure the Triton cache at import time as a best-effort convenience, but do
# NOT let a non-writable cache dir abort the import. On a locked-down deploy image
# (read-only home, no writable /triton-cache) configure_triton_cache_dir() raises
# RuntimeError; if that fired here unguarded it would shadow the HUD loud-fail
# (assert_templates_registered) with a confusing Triton error before any guard ran.
# The authoritative configuration happens in the @env.initialize serve hook, which
# fails loud there if it truly cannot find a cache dir. Here we only warn.
try:
    configure_triton_cache_dir()
except Exception as _triton_exc:  # pragma: no cover - exercised only on locked-down hosts.
    print(f"protean env: import-time Triton cache configuration skipped ({_triton_exc}); will retry at serve startup.")


def _read_prompt(op: str) -> str:
    spec = get_op(op)
    path = Path(__file__).resolve().parents[2] / spec.prompt_path
    return path.read_text()


def hud_prompt(op: str, split: str, shape: int | None = None) -> str:
    shape = shape or shapes_for_split(_as_split(split))[0]
    return (
        _read_prompt(op)
        + "\n\n"
        + "HUD grading metadata:\n"
        + f"- op: {op}\n"
        + f"- split: {split}\n"
        + f"- shape: {shape}\n"
        + "- reward info includes correctness, speedup, eager timing, kernel timing, and anti-hack caps.\n"
    )


def grade_hud_source(source: str, *, op: str, split: str, shape: int | None = None) -> dict[str, Any]:
    split_lit = _as_split(split)
    shape = shape or shapes_for_split(split_lit)[0]
    grade = grade_source(source or "", op=op, split=split_lit, shape=shape)
    return {
        **grade,
        "hud": {
            "task_id": f"{op}_{split}",
            "op": op,
            "split": split,
            "shape": shape,
            "reward": grade["reward"],
            "correct": grade["correct"],
            "speedup": grade["speedup"],
            "t_eager_ms": grade["t_eager_ms"],
            "t_kernel_ms": grade["t_kernel_ms"],
            "caps": grade["caps"],
        },
    }


def task_metadata() -> list[dict[str, Any]]:
    rows = []
    for task in HUD_TASKS:
        split = _as_split(task["split"])
        rows.append(
            {
                **task,
                "shape": shapes_for_split(split)[0],
                "prompt_path": get_op(task["op"]).prompt_path,
            }
        )
    return rows


def _make_env():
    if Environment is None:
        return None
    try:
        return Environment(name="protean")
    except TypeError:  # pragma: no cover - compatibility with older local HUD builds.
        return Environment(id="protean")


env = _make_env()


#: Template ids that MUST be registered for a healthy deploy. Kept as an explicit
#: tuple so the loud-fail guard and the initialize hook can verify against a single
#: source of truth rather than a magic number.
EXPECTED_TEMPLATE_IDS: tuple[str, ...] = (
    "elementwise_add_relu",
    "rmsnorm",
    "softmax_rows",
)


def registered_template_ids() -> tuple[str, ...]:
    """Return the template ids HUD actually registered on ``env``.

    Returns an empty tuple when ``hud`` is unavailable (``env is None``) so the
    hud-less import path stays non-fatal; the loud-fail logic lives in
    :func:`assert_templates_registered`.
    """

    if env is None:
        return ()
    # v6 exposes both ``tasks`` and ``templates`` dicts keyed by template id.
    registry = getattr(env, "tasks", None) or getattr(env, "templates", None) or {}
    try:
        return tuple(registry)
    except TypeError:  # pragma: no cover - defensive: unexpected registry shape.
        return ()


def assert_templates_registered() -> None:
    """Fail loud if the env is missing or any expected template did not register.

    This is the deploy/serve safety net: the guarded ``hud`` import keeps the
    package importable without the ``hud`` extra (CI's ``test`` extra omits it),
    but that same guard means a *failed* hud import in the deploy image would
    otherwise silently register ZERO templates and serve an empty env. Call this
    only from a serve/deploy context (the ``@env.initialize`` hook below, or the
    ``sys.argv``/``HUD_SERVE`` guard) so the hud-less import path is unaffected.
    """

    if env is None:
        raise RuntimeError(
            "Protean HUD env cannot serve: the 'hud' package is not importable "
            "(Environment is None), so zero templates are registered. Install "
            "hud-python[agents] in the deploy image (see Dockerfile.hud)."
        )
    registered = set(registered_template_ids())
    missing = [tid for tid in EXPECTED_TEMPLATE_IDS if tid not in registered]
    if missing:
        raise RuntimeError(
            "Protean HUD env cannot serve: env loaded but template registration "
            f"is incomplete. Missing {missing!r}; registered {sorted(registered)!r}. "
            "Expected elementwise_add_relu, rmsnorm, softmax_rows."
        )


def _template(template_id: str) -> Any:
    # The concrete return type is HUD's opaque template decorator; hud.* has no
    # stubs (ignore_missing_imports), so this is annotated as Any.
    assert env is not None
    try:
        return env.template(id=template_id)
    except TypeError:  # pragma: no cover - compatibility with older local HUD builds.
        return env.template(name=template_id)


if env is not None:

    @_template("elementwise_add_relu")
    async def elementwise_add_relu(split: str = "train", shape: int | None = None) -> AsyncGenerator[Any, str | None]:
        get_op("elementwise_add_relu")
        source = yield hud_prompt("elementwise_add_relu", split, shape)
        yield to_eval_result(grade_hud_source(source or "", op="elementwise_add_relu", split=split, shape=shape))

    @_template("rmsnorm")
    async def rmsnorm(split: str = "train", shape: int | None = None) -> AsyncGenerator[Any, str | None]:
        get_op("rmsnorm")
        source = yield hud_prompt("rmsnorm", split, shape)
        yield to_eval_result(grade_hud_source(source or "", op="rmsnorm", split=split, shape=shape))

    @_template("softmax_rows")
    async def softmax_rows(split: str = "train", shape: int | None = None) -> AsyncGenerator[Any, str | None]:
        get_op("softmax_rows")
        source = yield hud_prompt("softmax_rows", split, shape)
        yield to_eval_result(grade_hud_source(source or "", op="softmax_rows", split=split, shape=shape))

    elementwise_add_relu_train = elementwise_add_relu(split="train")
    elementwise_add_relu_train.slug = "elementwise_add_relu_train"
    elementwise_add_relu_train.columns = {"op": "elementwise_add_relu", "split": "train"}

    elementwise_add_relu_held_out = elementwise_add_relu(split="held_out")
    elementwise_add_relu_held_out.slug = "elementwise_add_relu_held_out"
    elementwise_add_relu_held_out.columns = {"op": "elementwise_add_relu", "split": "held_out"}

    rmsnorm_train = rmsnorm(split="train")
    rmsnorm_train.slug = "rmsnorm_train"
    rmsnorm_train.columns = {"op": "rmsnorm", "split": "train"}

    rmsnorm_held_out = rmsnorm(split="held_out")
    rmsnorm_held_out.slug = "rmsnorm_held_out"
    rmsnorm_held_out.columns = {"op": "rmsnorm", "split": "held_out"}

    softmax_rows_train = softmax_rows(split="train")
    softmax_rows_train.slug = "softmax_rows_train"
    softmax_rows_train.columns = {"op": "softmax_rows", "split": "train"}

    softmax_rows_held_out = softmax_rows(split="held_out")
    softmax_rows_held_out.slug = "softmax_rows_held_out"
    softmax_rows_held_out.columns = {"op": "softmax_rows", "split": "held_out"}

    @env.initialize
    async def _on_serve_start() -> None:
        """Serve-time startup hook (runs once before the control channel serves).

        Near-no-op: it (1) asserts every expected template registered so a broken
        deploy fails loud in the container logs instead of silently serving an
        empty env, and (2) authoritatively configures the Triton cache directory
        (the import-time call is best-effort/non-fatal; this is where a truly
        unwritable cache dir fails loud, and it does so *after* the template
        assertion so a registration failure is attributed correctly). CUDA is
        intentionally probed best-effort only (kernel grading needs a GPU, but the
        env must still boot on CPU-only hosts for smoke tests), so a missing GPU is
        logged, not fatal.
        """

        # Order matters: assert templates FIRST so a registration failure is the
        # reported error, then configure the cache (authoritative; may raise).
        assert_templates_registered()
        configure_triton_cache_dir()
        try:  # pragma: no cover - GPU probe is environment-specific.
            import torch

            if not torch.cuda.is_available():
                print("protean env: CUDA not available; kernel grading will run on CPU/eager.")
        except Exception as exc:  # pragma: no cover - torch optional at serve boot.
            print(f"protean env: CUDA probe skipped ({exc}).")

    @env.shutdown
    async def _on_serve_stop() -> None:
        """Serve-time shutdown hook (near-no-op; no long-lived resources to release)."""

        return None

else:
    elementwise_add_relu = {"id": "elementwise_add_relu", "op": "elementwise_add_relu"}
    rmsnorm = {"id": "rmsnorm", "op": "rmsnorm"}
    softmax_rows = {"id": "softmax_rows", "op": "softmax_rows"}
    for _task_def in HUD_TASKS:
        globals()[_task_def["id"]] = dict(_task_def)


# Backward-compatible metadata alias used by older imports. Keep it non-Task so
# HUD task discovery does not count it as a duplicate public task.
kernel_opt = {"id": "elementwise_add_relu_train", "op": "elementwise_add_relu", "split": "train"}


# --- Deploy/serve loud-fail guard ---------------------------------------------
# The guarded ``from hud import Environment`` import above keeps the package
# importable without the ``hud`` extra (CI's ``test`` extra does not install it),
# but that same guard means a *failed* hud import in the deploy image would
# silently register ZERO templates and serve an empty env. To prevent that
# silent-empty deploy while preserving the hud-less import path, fail loud only
# when this module is actually loaded as the ``hud serve``/``hud dev`` entrypoint
# (detected via ``sys.argv`` or the ``HUD_SERVE`` env-var). Under ``pytest``
# neither condition holds, so this is a no-op for the test suite.
_SERVE_VERBS = frozenset({"serve", "dev"})
_RUNNING_AS_SERVE = bool(os.environ.get("HUD_SERVE")) or any(_arg in _SERVE_VERBS for _arg in sys.argv[1:3])
if _RUNNING_AS_SERVE:
    # Belt-and-suspenders with the @env.initialize hook: fail at import time too,
    # in case the env is loaded as a serve entrypoint by a path that does not run
    # the initialize hooks.
    #
    # ORDERING ASSUMPTION: templates register at decorator-application time (the
    # @_template(...) decorators above run during module import), so by the time
    # this guard executes the templates are already registered or never will be.
    # That makes this import-time check correct for the current HUD SDK. If the
    # SDK ever switches to LAZY template registration (templates registered only
    # when the serve loop starts, after import), this guard would falsely pass on
    # zero templates -- in that case move it into @env.initialize only. The
    # test_templates_registered_immediately_after_import test below pins the
    # current eager-registration behavior and will fail if the SDK changes.
    assert_templates_registered()


def main() -> int:
    print(
        json.dumps(
            {
                "env_id": "protean",
                "hud_available": env is not None,
                "tasks": task_metadata(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
