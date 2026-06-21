from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from protean import rewards


def test_rewards_hash_placeholder_present_for_docker_bake():
    placeholder = "<sha256 baked at image build>"
    paths = [
        Path("src/protean/tasks/elementwise_add_relu/donotaccess/grade.py"),
        Path("src/protean/tasks/rmsnorm/donotaccess/grade.py"),
        Path("train/calibrate.py"),
    ]
    for path in paths:
        assert placeholder in path.read_text(), f"{path} is not hash-bakeable"


def test_dockerfile_bakes_hash_and_builds_donotaccess_moat():
    dockerfile = Path("Dockerfile.hud").read_text()
    assert "sha256sum /app/src/protean/rewards.py" in dockerfile
    assert "sed -i" in dockerfile
    assert "mkdir -p /donotaccess" in dockerfile
    assert "cp /app/src/protean/rewards.py /donotaccess/rewards.py" in dockerfile
    assert "cp /app/src/protean/reward_config.json /donotaccess/reward_config.json" in dockerfile
    assert "chmod -R 700 /donotaccess" in dockerfile
    assert 'su agent -c "cat /donotaccess/rewards.py"' in dockerfile


def test_production_donotaccess_permissions_when_present():
    if not Path("/donotaccess").exists():
        pytest.skip("/donotaccess only exists inside the built HUD image")
    assert oct(Path("/donotaccess").stat().st_mode & 0o777) == "0o700"
    assert Path("/donotaccess/rewards.py").is_file()
    assert Path("/donotaccess/reward_config.json").is_file()
    assert Path("/donotaccess/elementwise_add_relu/grade.py").is_file()
    assert Path("/donotaccess/rmsnorm/grade.py").is_file()


def test_triton_cache_dir_matches_dockerfile_and_env():
    dockerfile = Path("Dockerfile.hud").read_text()
    docker_match = re.search(r"^ENV\s+TRITON_CACHE_DIR=(\S+)", dockerfile, re.MULTILINE)
    assert docker_match is not None

    env_py = Path("src/protean/env.py").read_text()
    assert docker_match.group(1) == "/triton-cache"
    assert 'Path("/triton-cache")' in env_py
    assert "protean-triton" in env_py


def test_reward_config_loads_from_local_dev_copy(tmp_path, monkeypatch):
    cfg_path = tmp_path / "reward_config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "P_TARGET": 1.2,
                "SPEEDUP_FLOOR": 1.05,
                "SPEEDUP_CAP": 9.0,
                "CORRECT_FLOOR": 0.4,
                "SPEEDUP_REWARD_WEIGHT": 1.4,
                "PR_BONUS": 0.1,
                "MAX_REWARD": 1.8,
            }
        )
    )
    monkeypatch.setattr(rewards, "CANONICAL_CONFIG_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(rewards, "LOCAL_CONFIG_PATH", cfg_path)

    config = rewards.load_reward_config()
    assert config.p_target == 1.2
    assert config.speedup_floor == 1.05
    assert config.speedup_reward_weight == 1.4
    assert config.pr_bonus == 0.1


def test_pr_frac_moves_reward():
    base = dict(correct=True, speedup=1.5, launches_timed=1, dtype_ok=True, shape_ok=True, split="held_out")
    low = rewards.compute_reward(**base, pr_frac=0.1)["reward"]
    high = rewards.compute_reward(**base, pr_frac=0.9)["reward"]
    assert high > low
