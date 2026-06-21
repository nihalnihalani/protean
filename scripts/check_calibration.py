"""Pre-GRPO go/no-go gate. BLOCKS the 8 AM training kick unless the base model's
reward distribution is trainable. See IMPLEMENTATION_PLAN.md §4.7 / §6 (Block 3).

Asserts, over N rollouts/op of base Qwen2.5-Coder-7B via the REAL grader:
  - >= 5% compilable
  - >= 2% allclose (correct)
  - nonzero group reward variance (GRPO needs contrast)
  - median group reward in (0, 1)  → the 20–50% band
Exits nonzero on failure so a cron/Makefile won't proceed to GRPO.
"""
import sys
import os

# Set PYTHONPATH
local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(local_dir, "src"))
sys.path.insert(0, os.path.join(local_dir, "train"))

from protean.tasks import TASKS
from calibrate import calibrate

def main() -> int:
    print("Running calibration script...")
    res = calibrate(TASKS)
    if res in ("GO", "GO_ESCALATED"):
        print("GO: Calibration passed!")
        return 0
    else:
        print("FAIL: Calibration blocked!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
