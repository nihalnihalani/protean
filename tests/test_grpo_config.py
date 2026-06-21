from __future__ import annotations

from unittest.mock import patch

from train.grpo_loop import build_grpo_configs


def test_grpo_defaults_cpu(monkeypatch):
    monkeypatch.delenv("PROTEAN_MAX_STEPS", raising=False)
    monkeypatch.delenv("PROTEAN_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("PROTEAN_SMOKE_TEST", raising=False)
    monkeypatch.delenv("PROTEAN_DISABLE_VLLM", raising=False)

    with patch("train.grpo_loop._cuda_available", return_value=False):
        cfg, peft_cfg = build_grpo_configs()

    assert cfg.max_steps == 150
    assert cfg.num_generations == 8
    assert cfg.max_completion_length == 1024
    assert cfg.use_vllm is False
    assert cfg.vllm_gpu_memory_utilization == 0.0
    assert peft_cfg.target_modules == "all-linear"


def test_grpo_env_overrides(monkeypatch):
    monkeypatch.setenv("PROTEAN_MAX_STEPS", "42")
    monkeypatch.setenv("PROTEAN_OUTPUT_DIR", "/models/protean")
    monkeypatch.delenv("PROTEAN_SMOKE_TEST", raising=False)

    with patch("train.grpo_loop._cuda_available", return_value=False):
        cfg, _ = build_grpo_configs()

    assert cfg.max_steps == 42
    assert cfg.output_dir == "/models/protean"


def test_vllm_enabled_on_cuda(monkeypatch):
    monkeypatch.delenv("PROTEAN_SMOKE_TEST", raising=False)
    monkeypatch.delenv("PROTEAN_DISABLE_VLLM", raising=False)

    with patch("train.grpo_loop._cuda_available", return_value=True):
        cfg, _ = build_grpo_configs()

    assert cfg.use_vllm is True
    assert cfg.vllm_gpu_memory_utilization == 0.45


def test_vllm_disabled_for_smoke_test(monkeypatch):
    monkeypatch.setenv("PROTEAN_SMOKE_TEST", "1")

    with patch("train.grpo_loop._cuda_available", return_value=True):
        cfg, _ = build_grpo_configs()

    assert cfg.max_steps == 3
    assert cfg.num_generations == 2
    assert cfg.use_vllm is False
    assert cfg.vllm_gpu_memory_utilization == 0.0
