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

THRESHOLDS = dict(min_compilable=0.05, min_correct=0.02, band=(0.0, 1.0))


def main() -> int:
    # TODO: roll out base model N times per op through protean.grader.evaluate_kernel,
    # aggregate the metrics above, print a table, return 0 (pass) or 1 (block).
    raise NotImplementedError("see IMPLEMENTATION_PLAN.md §4.7")


if __name__ == "__main__":
    sys.exit(main())
