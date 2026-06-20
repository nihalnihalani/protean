"""Per-episode workspace staging. Writes the agent-facing prompt, a writable
solution.py stub, and a self-serve bench.py into /workdir each episode, and NEVER
copies the hidden reference/grader in. See IMPLEMENTATION_PLAN §4.2.
"""
import os

WORKSPACE_ROOT = os.environ.get("WORKSPACE_ROOT", "/workdir")


def hidden_dir(op_name: str) -> str:
    """Root:700 dir holding grade.py + reference.py for this op. Unreadable by the agent uid."""
    return os.path.join(os.path.dirname(__file__), "tasks", op_name, "donotaccess")


def setup_task(op_name: str, split: str, seed: int, validate_mode: str | None = None) -> dict:
    """Stage the workspace for one episode.

    TODO:
      - render prompt.md (shape NOT pinned — held-out) into the workspace
      - write /workdir/solution.py stub exposing `solution(*tensors)`
      - write /workdir/bench.py: agent self-check (compile + allclose + latency) on TRAIN shapes only
      - delete scripts/check_calibration.py from the agent-visible tree
      - return {"prompt": <str>, "seed": seed, "split": split}
    """
    raise NotImplementedError("see IMPLEMENTATION_PLAN.md §4.2")
