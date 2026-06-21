import re
from pathlib import Path

def test_triton_cache_dir_matches_dockerfile():
    dockerfile = Path("Dockerfile.hud").read_text()
    # Find the ENV TRITON_CACHE_DIR=... line
    m = re.search(r'^ENV\s+TRITON_CACHE_DIR=(\S+)', dockerfile, re.MULTILINE)
    assert m is not None, "TRITON_CACHE_DIR not set in Dockerfile.hud"
    dockerfile_value = m.group(1)
    
    # Find the Python-side setting in env.py
    env_py = Path("src/protean/env.py").read_text()
    m = re.search(r'"TRITON_CACHE_DIR":\s*"([^"]+)"', env_py)
    assert m is not None, "TRITON_CACHE_DIR not set in env.py _KernelWorkspace"
    python_value = m.group(1)
    
    assert dockerfile_value == python_value, (
        f"TRITON_CACHE_DIR mismatch: Dockerfile says {dockerfile_value!r}, "
        f"env.py says {python_value!r}. This wastes the JIT warmup."
    )
