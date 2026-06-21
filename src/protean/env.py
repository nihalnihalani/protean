"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

import os
import sys
from hud import Environment
from hud.environment import Workspace
from .scenario_helpers import WORKSPACE_ROOT, setup_task, _resolve_workspace_root
from .task_catalog import OPS_BY_NAME
from .grader import evaluate_kernel

AGENT_UID = int(os.environ.get("AGENT_UID", "1000"))
AGENT_GID = int(os.environ.get("AGENT_GID", "1000"))

class _KernelWorkspace(Workspace):
    # demote agent shell so it cannot read the root:700 hidden grader/reference
    def shell_argv(self, command=None, *, cwd=None, env=None) -> list[str]:
        argv = super().shell_argv(command, cwd=cwd, env=env)
        if sys.platform != "win32" and getattr(os, "geteuid", lambda: 1)() == 0:
            argv = ["setpriv", "--reuid", str(AGENT_UID), "--regid", str(AGENT_GID), "--clear-groups", "--", *argv]
        return argv

env = Environment(name="protean")
_ws = _KernelWorkspace(
    WORKSPACE_ROOT, 
    network=False,
    env={"HOME": "/home/agent", "USER": "agent", "TRITON_CACHE_DIR": "/triton-cache"}
)

@env.initialize
async def _up(): 
    await _ws.start()
    env.add_capability(_ws.capability("shell"))

@env.shutdown
async def _down(): 
    await _ws.stop()

@env.template(id="kernel_opt")
async def kernel_opt(op_name: str, split: str = "train", seed: int = 0, validate_mode: str | None = None):
    # setup the task workspace
    setup_meta = setup_task(op_name, split=split, seed=seed, validate_mode=validate_mode)
    op_spec = OPS_BY_NAME[op_name]
    
    # Expose prompt to agent
    prompt_path = os.path.join(os.path.dirname(__file__), op_spec.prompt_path)
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read()
    else:
        prompt = f"Optimize the {op_name} kernel on shape split {split}."

    answer = yield prompt
    
    # Grade the solution
    evaluation = evaluate_kernel(op_name=op_name, split=split, seed=seed)
    info = dict(evaluation.info or {})
    info["setup"] = setup_meta
    info["final_answer"] = None if answer is None else str(answer)
    evaluation.info = info
    
    yield evaluation
