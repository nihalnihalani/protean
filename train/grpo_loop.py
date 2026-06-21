"""GRPO training configuration for Protean stretch runs.

The verifier-first demo does not depend on GRPO. This module keeps the training
path importable on CPU/dev machines while resolving production values for CUDA
hosts where TRL and vLLM are installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from protean.manifest import load_frozen_manifest
from protean.sampler import l1_curriculum_pool, sample_task_curriculum
from protean.task_catalog import OPS_BY_NAME


ROOT = Path(__file__).resolve().parents[1]
STEP_REF = [0]


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


def build_dataset(*, step: int = 0, max_steps: int | None = None):
    """Build the GRPO dataset from the frozen manifest.

    Manifest rows preserve exact shapes. If dynamic manifest mode is enabled
    and no manifest exists, the dataset falls back to curriculum sampling.
    """

    allow_dynamic = os.environ.get("PROTEAN_ALLOW_DYNAMIC_MANIFEST") == "1"
    rows = load_frozen_manifest(allow_missing=allow_dynamic)
    max_steps = max_steps or int(os.environ.get("PROTEAN_MAX_STEPS", "150"))
    if not rows and allow_dynamic:
        rows = [
            sample_task_curriculum(op, idx, split, step=step, max_steps=max_steps)
            for op in OPS_BY_NAME
            for split in ("train", "held_out")
            for idx in range(3)
        ]

    data = {"prompt": [], "op": [], "shape": [], "dtype": [], "split": []}
    for row in rows:
        op = row["op"]
        op_spec = OPS_BY_NAME[op]
        prompt_path = ROOT / op_spec.prompt_path
        data["prompt"].append(prompt_path.read_text())
        data["op"].append(op)
        data["shape"].append(int(row["shape"]))
        data["dtype"].append(row.get("dtype", op_spec.dtype))
        data["split"].append(row["split"])

    try:
        from datasets import Dataset

        return Dataset.from_dict(data)
    except Exception:
        return [dict(zip(data.keys(), values)) for values in zip(*data.values())]


def _dataset_rows(dataset) -> list[dict]:
    if isinstance(dataset, list):
        return dataset
    return [dataset[i] for i in range(len(dataset))]


def _make_held_out_eval_fn(trainer, dataset):
    """Evaluate the current policy greedily on held-out manifest rows."""

    def eval_fn(step: int, n_per_op: int) -> tuple[float, float]:
        import torch
        from protean.grader import grade_source

        rows = [row for row in _dataset_rows(dataset) if row["split"] == "held_out"]
        sample = rows[: max(1, n_per_op) * max(1, len(OPS_BY_NAME))]
        rewards: list[float] = []
        tokenizer = getattr(trainer, "tokenizer", None) or getattr(trainer, "processing_class", None)
        if tokenizer is None:
            raise AttributeError("trainer has neither tokenizer nor processing_class")
        for row in sample:
            inputs = tokenizer(row["prompt"], return_tensors="pt").to(trainer.model.device)
            with torch.no_grad():
                output_ids = trainer.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            result = grade_source(
                completion,
                op=row["op"],
                split=row["split"],
                shape=int(row["shape"]),
                dtype=row["dtype"],
            )
            rewards.append(float(result.get("reward", 0.0)))
        return (float(mean(rewards)) if rewards else 0.0, float(pstdev(rewards)) if len(rewards) > 1 else 0.0)

    return eval_fn


def _make_base_model_runner(trainer):
    """Sample four completions per prompt for calibration reward-spread checks."""

    def runner(prompt: str) -> list[str]:
        import torch

        tokenizer = getattr(trainer, "tokenizer", None) or getattr(trainer, "processing_class", None)
        if tokenizer is None:
            raise AttributeError("trainer has neither tokenizer nor processing_class")
        inputs = tokenizer(prompt, return_tensors="pt").to(trainer.model.device)
        completions = []
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
            completions.append(tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))
        return completions

    return runner


def main() -> None:
    try:
        from train.calibrate import calibrate
        from train.callbacks import CostAbortCallback, RewardCurveLogger
    except Exception:
        from calibrate import calibrate
        from callbacks import CostAbortCallback, RewardCurveLogger

    cfg, peft_config = build_grpo_configs()
    dataset = build_dataset(step=0, max_steps=cfg.max_steps)
    calibrate(dataset if isinstance(dataset, list) else _dataset_rows(dataset), base_model_runner=None)
    history_path = os.path.join(cfg.output_dir, "train_history.json")
    callbacks = [
        CostAbortCallback(review_step=min(150, cfg.max_steps)),
        RewardCurveLogger(
            output_path=history_path,
            heldout_every=25,
            config_snapshot={
                "max_steps": cfg.max_steps,
                "num_generations": cfg.num_generations,
                "max_completion_length": cfg.max_completion_length,
                "learning_rate": cfg.learning_rate,
                "use_vllm": cfg.use_vllm,
            },
        ),
    ]
    print("[protean] GRPO config")
    for key in ("max_steps", "num_generations", "max_completion_length", "use_vllm", "vllm_gpu_memory_utilization"):
        print(f"{key}={getattr(cfg, key)}")
    print(f"initial_curriculum_pool={l1_curriculum_pool(0, cfg.max_steps)}")
    print(f"dataset_rows={len(dataset)}")
    print(f"lora_target_modules={peft_config.target_modules}")
    print(f"callbacks={[type(cb).__name__ for cb in callbacks]}")
    print("Training loop wiring is ready; install train extras to run TRL GRPO.")


if __name__ == "__main__":
    main()
