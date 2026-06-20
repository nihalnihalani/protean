"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# sampler.py — cross-machine determinism via sha256 of a canonical string (NOT builtin hash())
import hashlib, random
from protean.splits import TRAIN_M, TEST_M   # FIX (devil's-advocate R2): these were used unimported -> NameError

def _rng(op, idx, split):
    s = f"{op}|{idx}|{split}"
    seed = int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)
    return random.Random(seed)

def sample_task(op, idx, split):                # split in {"train","test"}
    pool = TRAIN_M if split == "train" else TEST_M
    r = _rng(op, idx, split)
    M = r.choice(pool); N = r.choice(pool)
    return dict(op=op, M=M, N=N, dtype="fp16", split=split)
