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
from callbacks import CostAbortCallback, RewardCurveLogger
from calibrate import calibrate
from protean.tasks import TASKS
from protean.sampler import l1_curriculum_pool
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

def build_dataset(step=0, max_steps=150):
    """Build the GRPO training dataset from the frozen manifest.
    
    Step 11: reads M/N directly from the manifest (via task.columns) when
    available, so the training shapes exactly match the frozen manifest_v1.jsonl.
    Falls back to curriculum-aware sampling only when the manifest doesn't
    include shape data (e.g. dynamic generation in local dev mode).
    
    Held-out tasks always use the full TEST_M pool (moat invariant).
    """
    data = {
        "prompt": [],
        "op": [],
        "M": [],
        "N": [],
        "dtype": [],
        "split": [],
    }
    
    for task in TASKS:
        op_name = task.columns["op"]
        split = task.columns["split"]
        seed = int(task.columns["seed"])
        
        # Prefer manifest M/N (step 11: reproducibility). Fall back to
        # curriculum-aware sampling if the manifest didn't include shapes
        # (e.g. dynamic generation via PROTEAN_ALLOW_DYNAMIC_MANIFEST=1).
        if "M" in task.columns and "N" in task.columns:
            M = int(task.columns["M"])
            N = int(task.columns["N"])
        else:
            M, N = sample_shape_curriculum(op_name, split, seed, step=step, max_steps=max_steps)
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
        data["split"].append(split)
        
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

def _make_held_out_eval_fn(trainer, dataset_heldout):
    """Build the held-out eval closure that the callback calls every N steps.
    
    Uses the trainer's current model (with LoRA weights applied at this step) to
    generate completions for held-out tasks, then grades them through the canonical
    grade_kernel path. Returns (mean_reward, std_reward) across all held-out rollouts.
    """
    from reward import _extract_code, _load_grade
    import numpy as np
    
    def eval_fn(step: int, n_per_op: int) -> tuple:
        # Sample n_per_op held-out tasks
        held_out_rows = [row for row in dataset_heldout if row["split"] == "held_out"]
        sample = held_out_rows[:n_per_op * 5]  # 5 ops; cap on total eval rollouts
        
        rewards = []
        for row in sample:
            op_name = row["op"]
            # Generate ONE completion for this held-out task — held-out is just a probe,
            # not a training rollout, so we don't need num_generations samples.
            prompt = row["prompt"]
            # Use trainer.model + tokenizer for inference. Inside vLLM-colocated trl,
            # the trainer exposes the underlying model.
            tokenizer = getattr(trainer, "tokenizer", None) or getattr(trainer, "processing_class", None)
            if tokenizer is None:
                raise AttributeError("Trainer has neither tokenizer nor processing_class")
            inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
            with torch.no_grad():
                output_ids = trainer.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,  # greedy for deterministic held-out
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # Grade via the canonical path
            try:
                gm = _load_grade(op_name)
                src = _extract_code(completion)
                res = gm.grade_kernel(
                    op_name, int(row["M"]), int(row["N"]), row["dtype"], src, step=step
                )
                rewards.append(float(res.get("reward", 0.0)))
            except Exception as e:
                print(f"[protean] Held-out grade failed for {op_name}: {e}")
                rewards.append(0.0)
        
        if not rewards:
            return (0.0, 0.0)
        return (float(np.mean(rewards)), float(np.std(rewards)))
    
    return eval_fn

def _make_base_model_runner(trainer):
    """Build the base-model rollout closure for calibrate() Stage A/B.
    
    Uses sampling (temperature=0.8, top_p=0.95) — NOT greedy — because the
    calibration gate needs to measure the *distribution* of model outputs
    (compile rate, allclose rate, reward std). Greedy would produce a single
    deterministic completion per prompt, making std measurements meaningless.
    
    Must be called AFTER trainer construction so we close over trainer.model.
    """
    def runner(prompt: str) -> list:
        tokenizer = getattr(trainer, "tokenizer", None) or getattr(trainer, "processing_class", None)
        if tokenizer is None:
            raise AttributeError("Trainer has neither tokenizer nor processing_class")
        inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
        
        completions = []
        # Generate 4 completions per prompt for calibration diversity
        for _ in range(4):
            with torch.no_grad():
                output_ids = trainer.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            completions.append(completion)
        return completions
    
    return runner

def main():
    # 1. Prepare dataset (L1 curriculum: step=0 uses smallest shapes)
    max_steps = int(os.environ.get("PROTEAN_MAX_STEPS", "150"))
    dataset = build_dataset(step=0, max_steps=max_steps)
    print(f"[protean] L1 curriculum: initial pool = {l1_curriculum_pool(0, max_steps)}")
    
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
    # Preflight only (no base_model_runner yet — we haven't built the trainer).
    # This validates the known-good/known-bad gate and REWARDS_HASH integrity.
    calibrate(TASKS, base_model_runner=None)
    
    # 3. Model path
    model_name = os.environ.get("SFT_CKPT_PATH", "Qwen/Qwen2.5-Coder-7B-Instruct")
    
    # 4. GRPO Trainer Setup
    cfg, peft_config = build_grpo_configs()
    
    # Snapshot the resolved config for the history file
    config_snapshot = {
        "max_steps": cfg.max_steps,
        "num_generations": cfg.num_generations,
        "max_completion_length": cfg.max_completion_length,
        "learning_rate": cfg.learning_rate,
        "use_vllm": cfg.use_vllm,
    }
    
    history_path = os.path.join(cfg.output_dir, "train_history.json")
    
    reward_logger = RewardCurveLogger(
        output_path=history_path,
        heldout_every=25,
        heldout_tasks_per_op=4,
        held_out_eval_fn=None,  # Wired AFTER trainer construction; see below
        config_snapshot=config_snapshot,
    )
    
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
        callbacks=[cost_callback, reward_logger],
        args=cfg,
        train_dataset=dataset,
    )
    
    # NOW we can build the eval function and base model runner with trainer in scope
    reward_logger.held_out_eval_fn = _make_held_out_eval_fn(trainer, dataset)
    
    # Step 9: Wire base-model rollouts into calibrate() for Stage A/B gating.
    # This runs AFTER trainer construction so _make_base_model_runner can close
    # over trainer.model. If Stage A/B fails, escalate() will lower reward targets.
    if os.environ.get("PROTEAN_SKIP_CALIBRATION") != "1":
        base_runner = _make_base_model_runner(trainer)
        cal_result = calibrate(TASKS, base_model_runner=base_runner)
        print(f"[protean] Calibration Stage A/B result: {cal_result}")
    
    class StepTrackerCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            STEP_REF[0] = state.global_step
            
    trainer.add_callback(StepTrackerCallback())
    
    print("Starting training loop...")
    trainer.train()

if __name__ == "__main__":
    main()
