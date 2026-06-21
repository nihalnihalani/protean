# Protean Demo Benchmark

Op: `elementwise_add_relu`. Base is PyTorch eager. Candidate is the hand-optimized Triton kernel.

| Split | Shape | Correct | Speedup | Eager ms | Triton ms | Reward | Caps |
|---|---:|---|---:|---:|---:|---:|---|
| train | 1024 | True | 1.563x | 0.005952 | 0.003808 | 0.481687 | - |
| train | 2048 | True | 1.563x | 0.005952 | 0.003808 | 0.481687 | - |
| train | 4096 | True | 1.563x | 0.005952 | 0.003808 | 0.481687 | - |
| held_out | 1536 | True | 2.071x | 0.007888 | 0.003808 | 0.627329 | - |
| held_out | 3072 | True | 1.563x | 0.005952 | 0.003808 | 0.481687 | - |
| held_out | 5632 | True | 2.076x | 0.007904 | 0.003808 | 0.628377 | - |
