import os
import pytest
from unittest.mock import patch
from grpo_loop import build_grpo_configs

def test_grpo_config_defaults(monkeypatch):
    # Ensure env variables are cleared/mocked
    monkeypatch.delenv("PROTEAN_MAX_STEPS", raising=False)
    monkeypatch.delenv("PROTEAN_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("PROTEAN_SMOKE_TEST", raising=False)
    monkeypatch.delenv("PROTEAN_DISABLE_VLLM", raising=False)
    
    cfg, peft_cfg = build_grpo_configs()
    
    assert cfg.max_steps == 150
    assert cfg.num_generations == 8
    assert cfg.max_completion_length == 1024
    assert peft_cfg.target_modules == "all-linear"
    assert cfg.output_dir == "./outputs"
    
    # On CPU-only dev hosts (the default on this test runner)
    assert cfg.use_vllm is False
    assert cfg.vllm_gpu_memory_utilization == 0.0

def test_grpo_config_overrides(monkeypatch):
    monkeypatch.setenv("PROTEAN_MAX_STEPS", "42")
    monkeypatch.setenv("PROTEAN_OUTPUT_DIR", "/models/grpo_run_123")
    monkeypatch.delenv("PROTEAN_SMOKE_TEST", raising=False)
    monkeypatch.delenv("PROTEAN_DISABLE_VLLM", raising=False)
    
    cfg, peft_cfg = build_grpo_configs()
    
    assert cfg.max_steps == 42
    assert cfg.output_dir == "/models/grpo_run_123"
    assert cfg.num_generations == 8
    assert peft_cfg.target_modules == "all-linear"

def test_grpo_config_smoke_test(monkeypatch):
    monkeypatch.delenv("PROTEAN_MAX_STEPS", raising=False)
    monkeypatch.setenv("PROTEAN_SMOKE_TEST", "1")
    monkeypatch.delenv("PROTEAN_DISABLE_VLLM", raising=False)
    
    cfg, peft_cfg = build_grpo_configs()
    
    assert cfg.max_steps == 3
    assert cfg.num_generations == 2
    assert cfg.save_steps == 1
    assert peft_cfg.target_modules == "all-linear"
    assert cfg.use_vllm is False
    assert cfg.vllm_gpu_memory_utilization == 0.0

def test_vllm_enabled_with_cuda(monkeypatch):
    monkeypatch.delenv("PROTEAN_MAX_STEPS", raising=False)
    monkeypatch.delenv("PROTEAN_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("PROTEAN_SMOKE_TEST", raising=False)
    monkeypatch.delenv("PROTEAN_DISABLE_VLLM", raising=False)
    
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            with patch("torch.cuda.set_device"):
                with patch("torch.cuda.current_device", return_value=0):
                    cfg, _ = build_grpo_configs()
                    assert cfg.use_vllm is True
                    assert cfg.vllm_gpu_memory_utilization == 0.45

def test_vllm_force_disabled_via_env(monkeypatch):
    monkeypatch.delenv("PROTEAN_MAX_STEPS", raising=False)
    monkeypatch.delenv("PROTEAN_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("PROTEAN_SMOKE_TEST", raising=False)
    monkeypatch.setenv("PROTEAN_DISABLE_VLLM", "1")
    
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            with patch("torch.cuda.set_device"):
                with patch("torch.cuda.current_device", return_value=0):
                    cfg, _ = build_grpo_configs()
                    assert cfg.use_vllm is False
                    assert cfg.vllm_gpu_memory_utilization == 0.0

def test_vllm_disabled_in_smoke_test_with_cuda(monkeypatch):
    monkeypatch.delenv("PROTEAN_MAX_STEPS", raising=False)
    monkeypatch.delenv("PROTEAN_OUTPUT_DIR", raising=False)
    monkeypatch.setenv("PROTEAN_SMOKE_TEST", "1")
    monkeypatch.delenv("PROTEAN_DISABLE_VLLM", raising=False)
    
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            with patch("torch.cuda.set_device"):
                with patch("torch.cuda.current_device", return_value=0):
                    cfg, _ = build_grpo_configs()
                    assert cfg.use_vllm is False
                    assert cfg.vllm_gpu_memory_utilization == 0.0


