import os
import sys
import modal

# Define Modal App
app = modal.App("protean")

# Build container image from the local Dockerfile
image = modal.Image.from_dockerfile("Dockerfile.hud")

# Persistent volume for model weights and checkpoints
models_volume = modal.Volume.from_name("protean-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="H100",
    timeout=1800, # 30 mins
    volumes={"/models": models_volume}
)
def smoke_test():
    """Verify H100 GPU and CUDA driver initialization inside the container."""
    import torch
    import triton
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"GPU Device: {device_name}")
        print(f"Device Capability: {capability}")
        return {
            "status": "ok",
            "cuda_available": True,
            "device_name": device_name,
            "capability": capability
        }
    return {"status": "error", "cuda_available": False}

@app.function(
    image=image,
    gpu="H100",
    timeout=600, # 10 mins
)
def bench_remote(payload: dict) -> dict:
    """
    Grade a single user Triton kernel inside the H100 sandbox.
    payload: {op, M, N, dtype, src}
    """
    sys.path.insert(0, "/mcp_server")
    from protean.subprocess_runner import run_bench
    
    op = payload["op"]
    M = payload["M"]
    N = payload["N"]
    dtype = payload["dtype"]
    src = payload["src"]
    
    res = run_bench(op, M, N, dtype, src, timeout_s=90)
    return res

@app.function(
    image=image,
    gpu="H100",
    timeout=86400, # 24 hours
    volumes={"/models": models_volume},
    secrets=[modal.Secret.from_name("hud-keys")]
)
def train():
    """Launch the overnight GRPO training loop on H100."""
    os.environ["SFT_CKPT_PATH"] = "/models/sft_warmup"
    os.environ["WORKSPACE_ROOT"] = "/workdir"
    
    sys.path.insert(0, "/mcp_server")
    sys.path.insert(0, "/mcp_server/train")
    
    import grpo_loop
    grpo_loop.main()

@app.function(
    image=image,
    gpu="H100",
    timeout=14400, # 4 hours
    volumes={"/models": models_volume}
)
def sft_warmup():
    """Launch the SFT warmup training loop on H100."""
    sys.path.insert(0, "/mcp_server")
    sys.path.insert(0, "/mcp_server/train")
    import sft_warmup
    sft_warmup.main()

@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    volumes={"/models": models_volume}
)
def eval_heldout(model_path: str = "/models/sft_warmup"):
    """Evaluate a trained model checkpoint against held-out test shapes."""
    sys.path.insert(0, "/mcp_server")
    from protean.tasks import TASKS
    from protean.grader import evaluate_kernel
    
    print(f"Evaluating model: {model_path} on held-out shapes...")
    # Generate scores and log generalization metrics
    results = []
    for task in TASKS:
        if task.columns["split"] == "held_out":
            op = task.columns["op"]
            seed = int(task.columns["seed"])
            print(f"Evaluating {op} (seed {seed}) on held-out shape...")
            res = evaluate_kernel(op, "held_out", seed)
            results.append(res)
            
    scores = [r.score for r in results]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    print(f"Held-out shape evaluation complete. Mean score: {mean_score:.3f}")
    return {"mean_score": mean_score, "results": [r.info for r in results]}
