# Protean Demo Benchmark

Base is PyTorch eager `relu(x + y)`. Candidate is the hand-optimized Triton kernel.

| Split | Shape | Correct | Speedup | Eager ms | Triton ms | Reward | Caps |
|---|---:|---|---:|---:|---:|---:|---|
| train | 1024 | True | 1.555x | 0.00592 | 0.003808 | 1.3 | - |
| train | 2048 | True | 1.559x | 0.005936 | 0.003808 | 1.3 | - |
| train | 4096 | True | 1.555x | 0.00592 | 0.003808 | 1.3 | - |
| held_out | 1536 | True | 2.076x | 0.007904 | 0.003808 | 1.3 | - |
| held_out | 3072 | True | 1.555x | 0.00592 | 0.003808 | 1.3 | - |
| held_out | 5632 | True | 2.076x | 0.007904 | 0.003808 | 1.3 | - |
