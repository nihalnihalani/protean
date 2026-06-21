"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# sampler.py — cross-machine determinism via sha256 of a canonical string (NOT builtin hash())
import hashlib, random

from .splits import TRAIN_M, TEST_M

def _rng(op, idx, split):
    s = f"{op}|{idx}|{split}"
    seed = int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)
    return random.Random(seed)

def sample_task(op, idx, split):                # split in {"train","test"}
    pool = TRAIN_M if split == "train" else TEST_M
    r = _rng(op, idx, split)
    M = r.choice(pool); N = r.choice(pool)
    return dict(op=op, M=M, N=N, dtype="fp16", split=split)

def sample_shape(op, split, seed):
    pool = TRAIN_M if split == "train" else TEST_M
    r = _rng(op, seed, split)
    M = r.choice(pool)
    N = r.choice(pool)
    return (M, N)


# ---------------------------------------------------------------------------
# L1 Shape Curriculum (Step 10)
# ---------------------------------------------------------------------------
# Progressive pool expansion: start with the smallest TRAIN_M shapes so the
# model gets easy wins before encountering hard tiling boundaries at 1024/2048.
#
# Schedule (fraction = step / max_steps):
#   [0.00, 0.15) → L1: 2 smallest shapes  (256, 320)
#   [0.15, 0.35) → L2: 4 shapes           (256, 320, 512, 640)
#   [0.35, 1.00] → L3: full TRAIN_M       (all 6)
#
# IMPORTANT: This only restricts the *training* pool. Held-out eval always
# uses the full TEST_M (the moat invariant is never touched — see splits.py).

# TRAIN_M is a tuple; we sort it to guarantee stable ordering for slicing.
_SORTED_TRAIN = tuple(sorted(TRAIN_M))

def l1_curriculum_pool(step: int, max_steps: int) -> tuple:
    """Return the active subset of TRAIN_M for the current training step.
    
    Args:
        step: Current global training step (0-indexed).
        max_steps: Total planned training steps (e.g. 150).
    
    Returns:
        A tuple of allowed M values, always a prefix-subset of sorted TRAIN_M.
    """
    if max_steps <= 0:
        return _SORTED_TRAIN  # safety: no curriculum if max_steps is bogus
    frac = step / max_steps
    if frac < 0.15:
        return _SORTED_TRAIN[:2]   # L1: (256, 320)
    elif frac < 0.35:
        return _SORTED_TRAIN[:4]   # L2: (256, 320, 512, 640)
    else:
        return _SORTED_TRAIN       # L3: full pool


def sample_shape_curriculum(op, split, seed, step=0, max_steps=150):
    """Like sample_shape but uses the L1 curriculum pool for training splits.
    
    For held-out / test splits, always uses the full TEST_M pool (moat invariant).
    """
    if split in ("train",):
        pool = l1_curriculum_pool(step, max_steps)
    else:
        pool = TEST_M
    r = _rng(op, seed, split)
    M = r.choice(pool)
    N = r.choice(pool)
    return (M, N)
