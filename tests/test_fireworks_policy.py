import json

from protean.model.fireworks_policy import _extract_json, build_kernel_edit_prompt


def test_extract_json_from_fenced_response():
    payload = {
        "name": "test_edit",
        "reason": "try a smaller block",
        "source": "import triton\n@triton.jit\ndef _k(): pass\ndef solution(x, y): return x",
    }
    text = "```json\n" + json.dumps(payload) + "\n```"
    assert _extract_json(text) == payload


def test_fireworks_prompt_keeps_kernel_contract():
    prompt = build_kernel_edit_prompt(
        op="elementwise_add_relu",
        current_best="def solution(x, y):\n    return x\n",
        best_summary={"mean_held_out_speedup": 1.5},
    )
    assert "Return ONLY JSON" in prompt
    assert "def solution" in prompt
    assert "do not call PyTorch" in prompt
