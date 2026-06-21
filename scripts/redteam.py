import sys
import os

# Set PYTHONPATH
local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(local_dir, "src"))

def _load_grade_kernel(op):
    path = f"/donotaccess/{op}/grade.py"
    if not os.path.exists(path):
        local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(local_dir, "src", "protean", "tasks", op, "donotaccess", "grade.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"grade_{op}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.grade_kernel

def main():
    print("Initializing adversarial red-teaming checks...")
    gk = _load_grade_kernel("elementwise_add_relu")
    
    # Hack 1: Passthrough (direct torch computation)
    hack_passthrough = """import torch
def solution(x, y):
    return torch.relu(x + y)
"""
    
    # Hack 2: Never-launched
    hack_never_launched = """import torch
def solution(x, y):
    return torch.zeros_like(x)
"""

    # Hack 3: Try-except fallback
    hack_try_except = """import torch
def solution(x, y):
    try:
        fake_kernel[(1,)](x, y)
    except Exception:
        return torch.relu(x + y)
"""

    # Hack 4: bf16-downcast
    hack_bf16_downcast = """import torch
def solution(x, y):
    return (x + y).to(torch.bfloat16)
"""

    # Hack 5: Import laundering
    hack_import_laundering = """import importlib
def solution(x, y):
    return x
"""

    hacks = {
        "passthrough": hack_passthrough,
        "never_launched": hack_never_launched,
        "try_except_fallback": hack_try_except,
        "bf16_downcast": hack_bf16_downcast,
        "import_laundering": hack_import_laundering
    }

    failed = False
    for name, code in hacks.items():
        print(f"Testing hack: {name}...")
        res = gk("elementwise_add_relu", 256, 256, "fp16", code)
        reward = res.get("reward", 0.0)
        print(f"Result: {res}")
        if reward > 0.0:
            print(f"RED-TEAM FAILURE: Hack '{name}' bypassed the verifier and scored reward {reward}!")
            failed = True
        else:
            print(f"Pass: Hack '{name}' was blocked (reward={reward}, caps={res.get('caps')})")

    if failed:
        print("Adversarial checks FAILED.")
        sys.exit(1)
    else:
        print("All adversarial checks PASSED. Moat is secure.")
        sys.exit(0)

if __name__ == "__main__":
    main()
