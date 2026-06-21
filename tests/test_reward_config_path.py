import os
import json
import pytest
from protean import rewards

def test_cfg_returns_correct_keys():
    # Test that _cfg() returns all 6 keys expected by compute_reward()
    cfg = rewards._cfg()
    expected_keys = {"P_TARGET", "SPEEDUP_FLOOR", "SPEEDUP_CAP", "CORRECT_FLOOR", "PR_BONUS", "TAU"}
    assert expected_keys.issubset(set(cfg.keys())), f"Missing keys in config: {expected_keys - set(cfg.keys())}"

def test_cfg_does_not_cache(tmp_path, monkeypatch):
    # Setup temporary config file
    temp_cfg_file = tmp_path / "reward_config.json"
    initial_data = {
        "P_TARGET": 1.5,
        "SPEEDUP_FLOOR": 1.1,
        "SPEEDUP_CAP": 20.0,
        "CORRECT_FLOOR": 0.3,
        "PR_BONUS": 0.2,
        "TAU": 0.5
    }
    
    with open(temp_cfg_file, "w") as f:
        json.dump(initial_data, f)
        
    # Monkeypatch the local config path
    monkeypatch.setattr(rewards, "_LOCAL_CFG_PATH", str(temp_cfg_file))
    
    # First call
    cfg1 = rewards._cfg()
    assert cfg1["P_TARGET"] == 1.5
    
    # Modify config file on disk
    initial_data["P_TARGET"] = 1.1
    with open(temp_cfg_file, "w") as f:
        json.dump(initial_data, f)
        
    # Second call (must hit disk and not be cached)
    cfg2 = rewards._cfg()
    assert cfg2["P_TARGET"] == 1.1
