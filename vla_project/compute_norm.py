#!/usr/bin/env python3
"""Compute normalization statistics (mean/std) from reconstructed dataset."""

import json, argparse
from pathlib import Path
import numpy as np


def compute(data_dir: Path, output_path: Path):
    consolidated = data_dir / "consolidated.npz"
    if not consolidated.exists():
        raise FileNotFoundError(f"{consolidated} not found – run reconstruct_dataset.py first")

    data = np.load(consolidated)
    states  = data["states"]   # [N, 14]
    actions = data["actions"]  # [N, 8]

    print(f"States:  {states.shape}  range=[{states.min():.3f}, {states.max():.3f}]")
    print(f"Actions: {actions.shape}  range=[{actions.min():.3f}, {actions.max():.3f}]")

    eps = 1e-8
    stats = {
        "norm_stats": {
            # State: arm(7) + effector(7) = 14
            "observation.state.arm.position": {
                "mean": states[:, :7].mean(axis=0).tolist(),
                "std":  np.maximum(states[:, :7].std(axis=0), eps).tolist(),
            },
            "observation.state.effector.position": {
                "mean": states[:, 7:14].mean(axis=0).tolist(),
                "std":  np.maximum(states[:, 7:14].std(axis=0), eps).tolist(),
            },
            # Action: arm(7) + effector(1) = 8
            "action.arm.position": {
                "mean": actions[:, :7].mean(axis=0).tolist(),
                "std":  np.maximum(actions[:, :7].std(axis=0), eps).tolist(),
            },
            "action.effector.position": {
                "mean": actions[:, 7:8].mean(axis=0).tolist(),
                "std":  np.maximum(actions[:, 7:8].std(axis=0), eps).tolist(),
            },
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(output_path, "w"), indent=2)
    print(f"\nNorm stats saved to: {output_path}")
    for k, v in stats["norm_stats"].items():
        m = [round(x, 3) for x in v["mean"]]
        s = [round(x, 3) for x in v["std"]]
        print(f"  {k}: mean={m}  std={s}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/data/coding/lingbot-vla/vla_project/reconstructed_data")
    p.add_argument("--output", default="/data/coding/lingbot-vla/assets/norm_stats/panda.json")
    args = p.parse_args()
    compute(Path(args.data_dir), Path(args.output))
