# Technical Specification

Protean is a verifier-first GPU-kernel optimization environment. A candidate kernel is useful only if it survives the same grading path locally, in the optimizer, and in HUD.

## 1. Public Contract

### Candidate Interface

Each candidate is a Python source string defining:

```python
@triton.jit
def _kernel(...):
    ...

def solution(...):
    ...
```

The public `solution(...)` signature depends on the op:

| Op | Signature | Reference |
|---|---|---|
| `elementwise_add_relu` | `solution(x, y)` | `torch.relu(x + y)` |
| `rmsnorm` | `solution(x, weight)` | `x * rsqrt(mean(x^2) + 1e-5) * weight` |

### Grader API

```python
grade_source(source, op="elementwise_add_relu", split="held_out", shape=None, reps=50, warmup=10)
```

Returns a dict with:

| Field | Meaning |
|---|---|
| `reward` | scalar reward |
| `correct` | `torch.allclose` against eager reference on fresh inputs |
| `speedup` | `t_eager_ms / t_kernel_ms` |
| `t_eager_ms` | PyTorch eager median CUDA-event time |
| `t_kernel_ms` | candidate median CUDA-event time |
| `split` | `train` or `held_out` |
| `shape` | shape used for grading |
| `caps` | anti-hack or failure reason list |
| `launches_timed` | timed Triton launches counted by the harness |
| `dtype_ok` | candidate dtype equals reference dtype |
| `shape_ok` | candidate shape equals reference shape |

## 2. Shape Split

Development split:

| Split | Shapes |
|---|---|
| train | `1024`, `2048`, `4096` |
| held-out | `1536`, `3072`, `5632` |

The sets are asserted disjoint at import time. `splits.py` also contains a future powered held-out sampler for larger evaluation, but the hackathon demo uses the fixed 3+3 dev split.

## 3. Verification Pipeline

```mermaid
flowchart TD
    A["candidate source"] --> B["ast_clean"]
    B -->|fail| Z["reward 0 + caps"]
    B -->|pass| C["contains_triton_jit"]
    C -->|fail| Z
    C -->|pass| D["CUDA available?"]
    D -->|no| Z
    D -->|yes| E["import source from temp .py"]
    E --> F["fresh inputs seed 42"]
    F --> G["candidate vs eager allclose"]
    G --> H["CUDA event timing seed 43"]
    H --> I["post timing correctness seed 44"]
    I --> J["compute_reward"]
```

The candidate is imported from a real temporary `.py` file because Triton JIT needs file-backed source.

## 4. Reward Formula

The current reward is intentionally simple:

```text
hard_failed = caps are non-empty
correctness_reward = 0.3 if correct and dtype_ok and shape_ok else 0.0
speedup_reward = min(speedup / 1.5, 1.0) if not hard_failed and speedup >= 1.1 else 0.0
reward = 0.0 if hard_failed else min(correctness_reward + speedup_reward, 2.0)
```

Current observed passing demo reward is `1.3` because:

```text
correctness_reward = 0.3
speedup_reward = 1.0
total = 1.3
```

This is not daVinci's multiplicative reward. It is Protean's v1 verifier reward: correct kernels get a floor; fast kernels get speedup credit; hard failures get zero.

## 5. Anti-Hack Gates

Static checks reject obvious bypasses:

| Gate | Example cap |
|---|---|
| banned PyTorch calls | `ast_ban:torch.relu` |
| no `@triton.jit` | `no_triton_jit` |
| dtype mismatch | `dtype_mismatch` |
| shape mismatch | `shape_mismatch` |
| incorrect output | `incorrect` |

Known red-team examples:

- PyTorch passthrough
- no launch / empty tensor
- bad shape Triton candidate

## 6. Optimizer Scoring

`optimizer.evaluate_kernel(...)` grades every train and held-out shape. Candidate summaries are scored as:

```python
(
    mean_held_out_speedup,
    mean_reward,
    correct_held_out,
)
```

Acceptance rule:

```python
candidate_score > best_score
```

This lexicographic order prioritizes held-out speed first, then average reward, then held-out correctness coverage.

## 7. Crash Safety

Model-generated kernels often fail to compile. Protean treats that as trace data, not a fatal run failure.

If candidate evaluation raises, the optimizer writes:

```json
{
  "accepted": false,
  "eval_error": {
    "type": "TypeError",
    "message": "grid must be a tuple"
  },
  "score": [0.0, 0.0, 0]
}
```

The source file remains under `runs/.../candidates/`.

## 8. HUD Adapter

HUD uses `EvaluationResult(reward=...)`. Protean's reward can exceed `1.0`, while HUD subscores must be `0..1`, so the adapter:

```python
EvaluationResult(
    reward=protean_reward,
    subscores=[SubScore(name="reward", value=min(protean_reward, 1.0), weight=1.0)],
    info=grade_dict,
)
```

The main HUD reward remains the Protean reward. The subscore is schema-safe metadata.

## 9. Verified Numbers

Spark GB10, HUD demo agent:

| Op | Split | Shape | Reward | Speedup |
|---|---|---:|---:|---:|
| `elementwise_add_relu` | train | 1024 | 1.3 | 1.55x |
| `elementwise_add_relu` | held-out | 1536 | 1.3 | 2.08x |
| `rmsnorm` | train | 1024 | 1.3 | 6.38x |
| `rmsnorm` | held-out | 1536 | 1.3 | 6.87x |

Passing HUD job:

https://hud.ai/jobs/813e572399c842c78d5a515f7644b4ae

## 10. Future Training Spec

The v1 learned layer is a 1,000,005-parameter policy head:

- input: 10 verifier-trace features
- hidden: 62,500
- output actions: block sizes `128`, `256`, `512`, `1024`, `2048`
- training signal: `delta_vs_best` from `trials.jsonl`

This is a controller over edit ordering, not a replacement for the coding model. A later 7B LoRA/GRPO run should be compared against this small controller and the deterministic baseline.
