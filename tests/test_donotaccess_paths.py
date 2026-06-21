import os
import pytest
import stat
from protean.grader import hidden_dir

def test_hidden_dir_resolves_locally():
    # Test that when running in source tree (no /donotaccess/),
    # hidden_dir() returns a path under the package directory (not /donotaccess/)
    p = hidden_dir("elementwise_add_relu")
    assert "/donotaccess/" not in p.replace("\\", "/")
    assert "tasks" in p
    assert "elementwise_add_relu" in p
    assert "donotaccess" in p
    
    # Test that the path returned by hidden_dir() exists and contains both grade.py and reference.py
    assert os.path.isdir(p), f"Path {p} is not a directory"
    assert os.path.isfile(os.path.join(p, "grade.py")), f"grade.py not found in {p}"
    assert os.path.isfile(os.path.join(p, "reference.py")), f"reference.py not found in {p}"

def test_donotaccess_production_permissions():
    # Skip checking /donotaccess/ permissions when not running inside the built image
    if not os.path.exists("/donotaccess"):
        pytest.skip("/donotaccess not found; skipping production image permissions check")
        
    # Check permissions of /donotaccess
    s = os.stat("/donotaccess")
    mode = s.st_mode & 0o777
    assert mode == 0o700, f"/donotaccess permissions are {oct(mode)}, expected 0700"
    
    # Check that canonical files exist
    assert os.path.isfile("/donotaccess/rewards.py"), "/donotaccess/rewards.py does not exist"
    assert os.path.isfile("/donotaccess/reward_config.json"), "/donotaccess/reward_config.json does not exist"
    assert os.path.isfile("/donotaccess/elementwise_add_relu/grade.py"), "elementwise_add_relu grade.py does not exist"
    assert os.path.isfile("/donotaccess/elementwise_add_relu/reference.py"), "elementwise_add_relu reference.py does not exist"
