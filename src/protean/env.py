"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# env.py  (skeleton)
import os
from hud import Environment
from hud.environment import Workspace
from scenario_helpers import WORKSPACE_ROOT, setup_task, _resolve_workspace_root
from task_catalog import OPS_BY_NAME

AGENT_UID = int(os.environ.get("AGENT_UID", "1000"))
AGENT_GID = int(os.environ.get("AGENT_GID", "1000"))

class _KernelWorkspace(Workspace):
    # mirror verilog uid-wall: setpriv to AGENT_UID, hidden /donotaccess root:700
    ...

@Environment.template(name="kernelforge")
def kernel_task(op: str, M: int, N: int, dtype: str = "fp16",
                split: str | None = "train"):    # plain str|None — NO Literal/Optional
    spec = OPS_BY_NAME[op]
    ws = setup_task(op, M, N, dtype, spec)        # writes prompt.md, copies hidden grade.py/reference.py
    prompt = ws.render_prompt()                    # KernelBench Model + get_inputs contract
    kernel_src = yield prompt                      # FIRST yield: agent returns a kernel

    workdir = _resolve_workspace_root()            # /workdir on image, per-pid tmp locally
    result = ws.grade(workdir, kernel_src)         # calls hidden grade.py via importlib
    yield result                                   # SECOND yield: EvaluationResult
