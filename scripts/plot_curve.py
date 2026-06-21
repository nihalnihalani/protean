"""Plot the real GRPO train-vs-held-out reward curve."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="outputs/train_history.json")
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        raise SystemExit(f"missing history file: {history_path}")
    data = json.loads(history_path.read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "curve_data.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "train_reward", "train_std", "heldout_reward", "heldout_std"])
        for i, step in enumerate(data["steps"]):
            writer.writerow(
                [
                    step,
                    "" if data["train_reward"][i] is None else data["train_reward"][i],
                    "" if data["train_std"][i] is None else data["train_std"][i],
                    "" if data["heldout_reward"][i] is None else data["heldout_reward"][i],
                    "" if data["heldout_std"][i] is None else data["heldout_std"][i],
                ]
            )

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"wrote: {csv_path}")
        print("matplotlib unavailable; skipped PNG")
        return 0

    def series(key: str, std_key: str):
        xs, ys, es = [], [], []
        for i, value in enumerate(data[key]):
            if value is not None:
                xs.append(data["steps"][i])
                ys.append(value)
                es.append(data[std_key][i] or 0.0)
        return xs, ys, es

    plt.figure(figsize=(10, 6))
    plt.errorbar(*series("train_reward", "train_std"), label="Train shapes", fmt="-o", capsize=4)
    plt.errorbar(*series("heldout_reward", "heldout_std"), label="Held-out shapes", fmt="--s", capsize=4)
    plt.title("Protean: Train vs Held-out Reward")
    plt.xlabel("GRPO step")
    plt.ylabel("Reward")
    plt.grid(True, linestyle=":")
    plt.legend()
    png_path = out_dir / "curve_train_vs_heldout.png"
    plt.savefig(png_path, dpi=200)
    print(f"wrote: {csv_path}")
    print(f"wrote: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
