"""GRPO training configuration for Protean stretch runs.

The verifier-first demo does not depend on GRPO. This module keeps the training
path importable on CPU/dev machines while resolving production values for CUDA
hosts where TRL and vLLM are installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LocalGRPOConfig:
    num_generations: int
    max_completion_length: int
    gradient_accumulation_steps: int
    use_vllm: bool
    vllm_gpu_memory_utilization: float
    learning_rate: float
    max_steps: int
    output_dir: str
    logging_steps: int
    save_steps: int
    save_total_limit: int
    report_to: str
    use_cpu: bool = False
    bf16: bool = True


@dataclass
class LocalLoraConfig:
    r: int
    lora_alpha: int
    target_modules: str
    bias: str
    task_type: str


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _build_local_config(*, use_vllm: bool, use_cpu: bool) -> LocalGRPOConfig:
    return LocalGRPOConfig(
        num_generations=8,
        max_completion_length=1024,
        gradient_accumulation_steps=4,
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=0.45 if use_vllm else 0.0,
        learning_rate=1e-5,
        max_steps=int(os.environ.get("PROTEAN_MAX_STEPS", "150")),
        output_dir=os.environ.get("PROTEAN_OUTPUT_DIR", "./outputs"),
        logging_steps=5,
        save_steps=25,
        save_total_limit=3,
        report_to="none",
        use_cpu=use_cpu,
        bf16=not use_cpu,
    )


def build_grpo_configs():
    """Return GRPO and LoRA configs with CUDA/vLLM-safe defaults."""

    smoke = os.environ.get("PROTEAN_SMOKE_TEST") == "1"
    cuda = _cuda_available()
    use_vllm = cuda and not smoke and os.environ.get("PROTEAN_DISABLE_VLLM") != "1"
    use_cpu = not cuda

    local_cfg = _build_local_config(use_vllm=use_vllm, use_cpu=use_cpu)
    if smoke:
        local_cfg.max_steps = 3
        local_cfg.num_generations = 2
        local_cfg.save_steps = 1
        local_cfg.use_vllm = False
        local_cfg.vllm_gpu_memory_utilization = 0.0

    local_lora = LocalLoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    try:
        from peft import LoraConfig
        from trl import GRPOConfig
    except Exception:
        return local_cfg, local_lora

    grpo_kwargs = local_cfg.__dict__.copy()
    cfg = GRPOConfig(**grpo_kwargs)
    peft_config = LoraConfig(**local_lora.__dict__)
    return cfg, peft_config


def main() -> None:
    try:
        from train.calibrate import calibrate
    except Exception:
        from calibrate import calibrate

    calibrate()
    cfg, peft_config = build_grpo_configs()
    print("[protean] GRPO config")
    for key in ("max_steps", "num_generations", "max_completion_length", "use_vllm", "vllm_gpu_memory_utilization"):
        print(f"{key}={getattr(cfg, key)}")
    print(f"lora_target_modules={peft_config.target_modules}")
    print("Training loop wiring is ready; install train extras to run TRL GRPO.")


if __name__ == "__main__":
    main()
