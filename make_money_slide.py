import json
import matplotlib.pyplot as plt
import sys
from pathlib import Path

candidates = [
    "reward_curve.json",
    "reward_curve.jsonl",
    "training_logs.txt",
    "outputs/reward_curve.json",
]

source = None
for c in candidates:
    if Path(c).exists():
        source = c
        break

if not source:
    print("ERROR: no curve data found. Tried:", candidates)
    sys.exit(1)

points = []
text = Path(source).read_text()
try:
    points = json.loads(text)
    if not isinstance(points, list):
        points = [points]
except json.JSONDecodeError:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                points.append(json.loads(line))
            except json.JSONDecodeError:
                pass

print(f"Loaded {len(points)} points from {source}")

train = [(p.get("step", i), p.get("reward", 0)) for i, p in enumerate(points) if p.get("split") == "train"]
heldout = [(p.get("step", i), p.get("reward", 0)) for i, p in enumerate(points) if p.get("split") in ("held_out", "heldout")]

plt.figure(figsize=(10, 6))
if train:
    plt.plot(*zip(*train), label=f"Train shapes (n={len(train)})", marker="o", linewidth=2)
if heldout:
    plt.plot(*zip(*heldout), label=f"Held-out off-grid shapes (n={len(heldout)})", marker="s", linewidth=2)
plt.xlabel("Training step")
plt.ylabel("Mean reward")
plt.title("Protean: Shape-Generalization on Triton Kernels")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("money_slide.png", dpi=150, bbox_inches="tight")
print("Saved money_slide.png")
