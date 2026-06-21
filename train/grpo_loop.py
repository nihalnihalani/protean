"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# grpo_loop.py — single-turn primary path
import os
import sys

# Get package root
local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(local_dir, "src")
if os.path.exists(src_dir):
    sys.path.insert(0, src_dir)
else:
    sys.path.insert(0, local_dir)
sys.path.insert(0, os.path.join(local_dir, "train"))

import torch
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig
from datasets import Dataset
from transformers import TrainerCallback

from reward import _extract_code, _load_grade
from callbacks import CostAbortCallback
from calibrate import calibrate
from protean.tasks import TASKS
from protean.shape_sampler import sample_shape
from protean.task_catalog import OPS_BY_NAME

# Reference step counter to allow dynamic reward adjustments if needed
STEP_REF = [0]

def dynamic_reward_fn(prompts, completions, op, M, N, dtype, **kwargs):
    rewards = []
    # completions are batched. For each element, we load the grader for that op and evaluate
    for i in range(len(prompts)):
        op_name = op[i]
        c = completions[i]
        
        # Extract code from the completion
        src = _extract_code(c)
        
        try:
            gm = _load_grade(op_name)
            res = gm.grade_kernel(
                op_name, 
                int(M[i]), 
                int(N[i]), 
                dtype[i], 
                src, 
                step=STEP_REF[0]
            )
            rewards.append(float(res.get("reward", 0.0)))
        except Exception as e:
            print(f"Error in reward function: {e}")
            rewards.append(0.0)
            
    return rewards

def build_dataset():
    data = {
        "prompt": [],
        "op": [],
        "M": [],
        "N": [],
        "dtype": [],
    }
    
    for task in TASKS:
        op_name = task.columns["op"]
        split = task.columns["split"]
        seed = int(task.columns["seed"])
        
        M, N = sample_shape(op_name, split, seed)
        op_spec = OPS_BY_NAME[op_name]
        
        # Read the prompt template
        prompt_path = os.path.join(local_dir, "src", "protean", op_spec.prompt_path)
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_content = f.read()
            
        data["prompt"].append(prompt_content)
        data["op"].append(op_name)
        data["M"].append(M)
        data["N"].append(N)
        data["dtype"].append("fp16")
        
    return Dataset.from_dict(data)

def build_grpo_configs():
    """Build the GRPOConfig and LoraConfig for training.
    
    vLLM resolution:
      - Enabled by default when CUDA is available (the production path).
      - Disabled when CUDA is unavailable (CPU-only dev hosts).
      - Force-disabled when PROTEAN_DISABLE_VLLM=1 (emergency override).
      - Force-disabled in smoke-test mode (PROTEAN_SMOKE_TEST=1) to avoid
        the ~30s vLLM engine init cost during quick wiring tests.
    
    When vLLM is disabled, vllm_gpu_memory_utilization is set to 0.0 to
    prevent trl from allocating any GPU memory for the (unused) vLLM engine.
    
    Plan §4.7: colocate vLLM with trainer at gpu_memory_utilization=0.45,
    leaving 55% of 80GB H100 memory for trainer weights + gradients + optimizer.
    """
    import os
    import torch
    extra_kwargs = {}
    if not torch.cuda.is_available():
        extra_kwargs["use_cpu"] = True
        extra_kwargs["bf16"] = False
        
    # vLLM requires CUDA. On CPU-only dev hosts, fall back to HF generate.
    # Plan §4.7: colocate vLLM with trainer at gpu_memory_utilization=0.45.
    use_vllm_resolved = torch.cuda.is_available() and os.environ.get("PROTEAN_DISABLE_VLLM") != "1"
    
    cfg = GRPOConfig(
        num_generations=8,
        max_completion_length=1024,
        gradient_accumulation_steps=4,
        use_vllm=use_vllm_resolved,
        vllm_gpu_memory_utilization=0.45 if use_vllm_resolved else 0.0,
        learning_rate=1e-5,
        max_steps=int(os.environ.get("PROTEAN_MAX_STEPS", "150")),
        output_dir=os.environ.get("PROTEAN_OUTPUT_DIR", "./outputs"),
        logging_steps=5,
        save_steps=25,
        save_total_limit=3,
        report_to="none",
        **extra_kwargs
    )
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",             # plan: all-linear, not just q/k/v/o
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    if os.environ.get("PROTEAN_SMOKE_TEST") == "1":
        cfg.max_steps = 3
        cfg.num_generations = 2
        cfg.save_steps = 1
        cfg.use_vllm = False
        cfg.vllm_gpu_memory_utilization = 0.0
        print("[protean] SMOKE TEST MODE: max_steps=3, num_generations=2, vllm disabled")
        
    return cfg, peft_config

def main():
    # 1. Prepare dataset
    dataset = build_dataset()
    
    # 2. Run calibration
    print("Running calibration preflight...")
    import os
    if hasattr(os, "geteuid") and os.geteuid() != 0 and os.path.exists("/donotaccess"):
        raise RuntimeError(
            "Trainer must run as root to write /donotaccess/reward_config.json during calibration "
            "escalation. Current euid: %d. If you intentionally demoted the trainer, you must "
            "also make /donotaccess/reward_config.json writable by that uid, or escalation will "
            "silently fail and the safety valve is gone." % os.geteuid()
        )
    if torch.cuda.is_available() and os.environ.get("PROTEAN_DISABLE_VLLM") == "1":
        print("[protean] WARNING: CUDA is available but vLLM is force-disabled via PROTEAN_DISABLE_VLLM.")
        print("[protean] WARNING: Training throughput will be 5-10× lower. This is fine for debugging but")
        print("[protean] WARNING: NOT suitable for the overnight GRPO kick.")
    calibrate(TASKS)
    
    # 3. Model path
    model_name = os.environ.get("SFT_CKPT_PATH", "Qwen/Qwen2.5-Coder-7B-Instruct")
    
    # 4. GRPO Trainer Setup
    cfg, peft_config = build_grpo_configs()
    
    print(f"[protean] GRPO config:")
    print(f"  max_steps           = {cfg.max_steps}")
    print(f"  num_generations     = {cfg.num_generations}")
    print(f"  max_completion_len  = {cfg.max_completion_length}")
    print(f"  learning_rate       = {cfg.learning_rate}")
    print(f"  output_dir          = {cfg.output_dir}")
    print(f"  use_vllm            = {cfg.use_vllm}")
    print(f"  vllm_gpu_mem_util   = {cfg.vllm_gpu_memory_utilization}")
    print(f"  cuda_available      = {torch.cuda.is_available()}")
    print(f"  lora target_modules = {peft_config.target_modules}")
    
    # Callback
    cost_callback = CostAbortCallback()
    
    print(f"Initializing GRPOTrainer with model: {model_name}")
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[dynamic_reward_fn],
        peft_config=peft_config,
        callbacks=[cost_callback],
        args=cfg,
        train_dataset=dataset,
    )
    
    class StepTrackerCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            STEP_REF[0] = state.global_step
            
    trainer.add_callback(StepTrackerCallback())
    
    print("Starting training loop...")
    trainer.train()

if __name__ == "__main__":
    main()
