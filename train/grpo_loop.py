"""Protean — stub extracted from docs/IMPLEMENTATION_PLAN.md (section 4). Fill in TODOs to implement."""

# grpo_loop.py — single-turn primary path
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

cfg = GRPOConfig(
    num_generations=8,                # → 4 if OOM
    max_completion_length=1024,
    gradient_accumulation_steps=4,
    use_vllm=True, vllm_gpu_memory_utilization=0.45,
    learning_rate=1e-5, max_steps=300,
)
trainer = GRPOTrainer(
    model=SFT_CKPT_PATH,              # SFT-warmed, NOT raw base
    reward_funcs=[reward_fn],         # the ONLY reward path; no LLM judge
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear"),
    callbacks=[RewardCurveLogger(heldout_every=25), CostAbortCallback()],
    args=cfg,
)
