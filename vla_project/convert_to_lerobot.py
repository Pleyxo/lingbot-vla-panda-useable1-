#!/usr/bin/env python3
"""
Convert consolidated.npz to LeRobot-compatible dataset format.
===========================================================================
Generates mp4 video files + parquet data following LeRobot format spec:
  lerobot_data/
  ├── meta/
  │   ├── info.json           # dataset metadata (data_path, video_path, features)
  │   ├── episodes.jsonl      # per-episode metadata
  │   ├── tasks.jsonl         # task descriptions
  │   └── stats.json          # normalization stats
  ├── data/
  │   └── chunk-000/
  │       ├── episode_000000.parquet   # state + action (no image)
  │       └── ...
  └── videos/
      └── chunk-000/
          └── observation.images.agentview/
              ├── episode_000000.mp4   # camera images
              └── ...

Usage:  python convert_to_lerobot.py
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────
RECON_DIR   = Path("/data/coding/lingbot-vla/vla_project/reconstructed_data")
NORM_PATH   = Path("/data/coding/lingbot-vla/assets/norm_stats/panda.json")
OUTPUT_DIR  = Path("/data/coding/lingbot-vla/vla_project/lerobot_data")
IMG_SIZE    = 224
FPS         = 20
TASK        = "pick up the red cube and lift it"

# LeRobot format path templates
DATA_PATH  = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def encode_episode_video(frames: np.ndarray, output_path: Path, fps: int):
    """Encode numpy frames [T, H, W, C] uint8 to mp4 using PyAV."""
    import av
    T, H, W, C = frames.shape
    assert C == 3, f"Expected 3-channel RGB, got {C}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23"}  # good quality/size tradeoff

    for i in range(T):
        av_frame = av.VideoFrame.from_ndarray(frames[i], format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    # Flush remaining packets
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = p.parse_args()
    out = Path(args.output)

    # ── Load consolidated data ──
    consolidated = RECON_DIR / "consolidated.npz"
    if not consolidated.exists():
        print(f"[ERROR] {consolidated} not found. Run reconstruct_dataset.py first.")
        return

    data = np.load(consolidated)
    images   = data["images"]    # [N, 224, 224, 3] uint8
    states   = data["states"]    # [N, 14] float32
    actions  = data["actions"]   # [N, 8] float32
    boundaries = data["episode_boundaries"]  # [ep+1]
    n_frames = len(images)
    n_eps    = len(boundaries) - 1

    print(f"Loaded: {n_frames} frames, {n_eps} episodes")
    print(f"  images:  {images.shape} {images.dtype}")
    print(f"  states:  {states.shape} {states.dtype}")
    print(f"  actions: {actions.shape} {actions.dtype}")

    # ── Load norm stats ──
    if NORM_PATH.exists():
        norm_data = json.loads(NORM_PATH.read_text())
        stats = norm_data["norm_stats"]
    else:
        stats = {}

    # ── Create directory structure ──
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    # ── Write meta/info.json ──
    info = {
        "codebase_version": "v3.0",
        "robot_type": "franka_panda",
        "fps": FPS,
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [8]},
            "observation.images.agentview": {
                "dtype": "video",
                "shape": [3, IMG_SIZE, IMG_SIZE],
            },
        },
        "total_episodes": n_eps,
        "total_frames": n_frames,
        "splits": {"train": f"0:{n_eps}"},
    }
    json.dump(info, (out / "meta" / "info.json").open("w"), indent=2)

    # ── Write meta/episodes.jsonl ──
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        for i in range(n_eps):
            start = int(boundaries[i])
            end   = int(boundaries[i + 1])
            ep = {"episode_index": i, "tasks": [TASK], "length": end - start}
            f.write(json.dumps(ep) + "\n")

    # ── Write meta/tasks.jsonl ──
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK}) + "\n")

    # ── Write meta/stats.json ──
    if stats:
        json.dump(stats, (out / "meta" / "stats.json").open("w"), indent=2)

    # ── Generate mp4 videos + parquet files per episode ──
    video_key = "observation.images.agentview"
    print(f"\nGenerating {n_eps} videos + parquet files ...")

    for ep_idx in range(n_eps):
        start = int(boundaries[ep_idx])
        end   = int(boundaries[ep_idx + 1])
        length = end - start

        # ── Encode video ──
        video_path = out / "videos" / "chunk-000" / video_key / f"episode_{ep_idx:06d}.mp4"
        ep_images = images[start:end]  # [T, H, W, C] uint8
        print(f"  episode {ep_idx+1}/{n_eps}: {length} frames → mp4 ...", end=" ", flush=True)
        encode_episode_video(ep_images, video_path, FPS)
        size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"{size_mb:.1f} MB")

        # ── Write parquet (NO image column — loaded from video) ──
        rows = []
        for frame_idx in range(length):
            global_idx = start + frame_idx
            rows.append({
                "observation.state": states[global_idx].tolist(),
                "action": actions[global_idx].tolist(),
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "timestamp": float(frame_idx) / FPS,
                "task_index": 0,
                "task": TASK,
                "index": global_idx,
                "action_is_pad": False,
            })

        df = pd.DataFrame(rows)
        parquet_path = out / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(parquet_path, index=False)

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"  LeRobot dataset ready: {out}")
    print(f"  Episodes: {n_eps}  |  Frames: {n_frames}")
    print(f"  State dim: 14  |  Action dim: 8  |  Image: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Next: update train_panda.yaml → train_path: {out}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
