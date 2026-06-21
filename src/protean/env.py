"""HUD public wrapper around the direct Protean verifier."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from protean.grader import grade_source, to_eval_result
from protean.splits import shapes_for_split
from protean.task_catalog import OPS, get_op

try:
    from hud import Environment
except Exception:  # pragma: no cover - local tests should not require HUD.
    Environment = None


HUD_TASKS = tuple(
    {"id": f"{op.name}_{split}", "op": op.name, "split": split}
    for op in OPS
    for split in ("train", "held_out")
)


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


configure_triton_cache_dir()


def _read_prompt(op: str) -> str:
    spec = get_op(op)
    path = Path(__file__).resolve().parents[2] / spec.prompt_path
    return path.read_text()


def hud_prompt(op: str, split: str, shape: int | None = None) -> str:
    shape = shape or shapes_for_split(split)[0]
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
    shape = shape or shapes_for_split(split)[0]
    grade = grade_source(source or "", op=op, split=split, shape=shape)
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
        split = task["split"]
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


def _template(template_id: str):
    try:
        return env.template(id=template_id)
    except TypeError:  # pragma: no cover - compatibility with older local HUD builds.
        return env.template(name=template_id)


if env is not None:

    def _register_op_template(op_name: str):
        @_template(op_name)
        async def _op_template(split: str = "train", shape: int | None = None):
            get_op(op_name)
            source = yield hud_prompt(op_name, split, shape)
            yield to_eval_result(grade_hud_source(source or "", op=op_name, split=split, shape=shape))

        _op_template.__name__ = op_name
        _op_template.__qualname__ = op_name
        return _op_template

    for _op_spec in OPS:
        _template_fn = _register_op_template(_op_spec.name)
        globals()[_op_spec.name] = _template_fn
        for _split in ("train", "held_out"):
            _task_slug = f"{_op_spec.name}_{_split}"
            _task = _template_fn(split=_split)
            _task.slug = _task_slug
            _task.columns = {"op": _op_spec.name, "split": _split}
            globals()[_task_slug] = _task
else:
    for _op_spec in OPS:
        globals()[_op_spec.name] = {"id": _op_spec.name, "op": _op_spec.name}
    for _task_def in HUD_TASKS:
        globals()[_task_def["id"]] = dict(_task_def)


# Backward-compatible metadata alias used by older imports. Keep it non-Task so
# HUD task discovery does not count it as a duplicate public task.
kernel_opt = {"id": "elementwise_add_relu_train", "op": "elementwise_add_relu", "split": "train"}


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
