#!/usr/bin/env python3
"""
数据重建脚本：从遥操录制的 action 序列重建完整训练数据
=========================================================================
原理：MuJoCo 确定性仿真 → 同样的 action 序列 → 完全一样的仿真过程
因此可从录制的 action 离线重建所有观测（图像+状态）

输出:
  reconstructed_data/
  ├── ep_0/step_0000.npz    # {image, state, action}
  ├── meta.json              # 数据集元信息
  └── consolidated.npz       # 所有数据合并

State dim:  14 = joint_pos_cos(7) + eef_pos(3) + gripper_qpos(1) + cube_pos(3)
Action dim: 8  = joint_velocity(7) + gripper_cmd(1)
Image:      agentview camera @ 224x224

任务: PickPlaceBox — 抓取红色方块并放入旁边的开口盒子中
"""

import os, sys, json, argparse
from pathlib import Path
import numpy as np

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"

# 导入自定义 PickPlaceBox 环境
sys.path.insert(0, str(Path(__file__).parent))
from pick_place_box import PickPlaceBox

# ── Paths ───────────────────────────────────────────────────────────
TELEOP_DIR  = Path("/data/coding/lingbot-vla/vla_project/teleop_data")
OUTPUT_DIR  = Path("/data/coding/lingbot-vla/vla_project/reconstructed_data")
LANGUAGE    = "pick up the red cube and place it into the open box"

# ── State layout (14 dim, matching model input shape) ───────────────
# joint_pos_cos(7) + eef_pos(3) + gripper_qpos(1) + cube_pos(3) = 14
STATE_KEYS  = [
    "robot0_joint_pos_cos",   # 7
    "robot0_eef_pos",         # 3
    "robot0_gripper_qpos",    # 1
    "cube_pos",               # 3
]  # total: 14 dim
ACTION_DIM  = 8   # 7 joint vel + 1 gripper


def build_state(obs: dict) -> np.ndarray:
    """Extract 14-dim state vector from robosuite observation.
    gripper_qpos is (2,) for two fingers; we take only the first."""
    parts = []
    for k in STATE_KEYS:
        v = obs[k].flatten()
        if k == "robot0_gripper_qpos":
            v = v[:1]  # take first finger only → 1 dim
        parts.append(v)
    return np.concatenate(parts).astype(np.float32)


def render(env, camera: str, size: int) -> np.ndarray:
    """Render camera image, flipped upright."""
    img = env.sim.render(width=size, height=size, camera_name=camera)
    return np.flipud(img).astype(np.uint8)


def reconstruct_one(env, trajectory: list, ep_idx: int,
                    camera: str, img_size: int) -> dict:
    """Replay one episode, save (image, state, action) per step."""
    ep_dir = OUTPUT_DIR / f"ep_{ep_idx}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    obs   = env.reset()
    total_reward = 0.0
    kept  = 0
    done  = False

    # Filter idle frames (all-zero action)
    active = [(i, t) for i, t in enumerate(trajectory)
              if not all(abs(a) < 1e-6 for a in t["action"])]
    total  = len(active)
    print(f"  Episode {ep_idx}: {total} active steps")

    for step_idx, (orig_idx, entry) in enumerate(active):
        action = np.array(entry["action"], dtype=np.float64)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)

        np.savez_compressed(
            ep_dir / f"step_{step_idx:05d}.npz",
            image=render(env, camera, img_size),
            state=build_state(obs),
            action=action.astype(np.float32),
        )
        kept += 1

        if done:
            print(f"    ✓ task completed at step {orig_idx}")
            break
        if (step_idx + 1) % 300 == 0:
            print(f"    ... {step_idx + 1}/{total}")

    return dict(episode=ep_idx, total_steps=kept,
                total_reward=float(total_reward), success=bool(done))


def make_meta(ep_metas: list, camera: str, img_size: int) -> dict:
    """Build LeRobot-compatible dataset metadata."""
    frames = sum(m["total_steps"] for m in ep_metas)
    episodes, offset = [], 0
    for m in ep_metas:
        episodes.append(dict(
            episode_index=m["episode"], tasks=[LANGUAGE],
            length=m["total_steps"],
            dataset_from_index=offset, dataset_to_index=offset + m["total_steps"]))
        offset += m["total_steps"]
    return dict(
        robot="Franka Panda (single-arm, 7-DOF + 1 gripper)",
        task="PickPlaceBox", camera=camera, img_size=img_size,
        state_dim=14, action_dim=ACTION_DIM, fps=20,
        total_episodes=len(ep_metas), total_frames=frames,
        language_task=LANGUAGE, episodes=episodes)


def main():
    p = argparse.ArgumentParser(description="Reconstruct VLA training data")
    p.add_argument("--episodes", type=str, default=None)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--camera", type=str, default="agentview")
    args = p.parse_args()

    files = sorted(TELEOP_DIR.glob("ep_*.json"))
    if not files:
        print(f"[ERROR] No ep_*.json in {TELEOP_DIR}"); sys.exit(1)
    if args.episodes:
        wanted = set(int(x.strip()) for x in args.episodes.split(","))
        files  = [f for f in files if int(f.stem.split("_")[1]) in wanted]

    print("=" * 55)
    print("  Panda VLA – PickPlaceBox Data Reconstruction")
    print("=" * 55)
    print(f"  Task:       Pick up cube and place into open box")
    print(f"  Input:      {TELEOP_DIR}  ({len(files)} episodes)")
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"  Camera:     {args.camera} @ {args.img_size}x{args.img_size}")
    print(f"  State dim:  14 = joint_cos(7)+eef(3)+gripper(1)+cube(3)")
    print(f"  Action dim: 8  = joint_vel(7)+gripper_cmd(1)")

    print("\n[1/3] Creating PickPlaceBox environment ...")
    env = PickPlaceBox(robots="Panda", has_renderer=False, has_offscreen_renderer=True,
                       use_camera_obs=False, control_freq=20, horizon=10000,
                       reward_shaping=True)
    cams = list(env.sim.model.camera_names)
    print(f"  DOF={env.robots[0].dof}  |  cameras={cams}")

    print(f"\n[2/3] Reconstructing ...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metas = []
    for i, fp in enumerate(files):
        data = json.loads(fp.read_text())
        m = reconstruct_one(env, data["trajectory"], i, args.camera, args.img_size)
        metas.append(m)
        print(f"    -> {m['total_steps']} steps | reward={m['total_reward']:.1f} | success={m['success']}")
    env.close()

    print(f"\n[3/3] Building consolidated dataset ...")
    meta = make_meta(metas, args.camera, args.img_size)
    json.dump(meta, (OUTPUT_DIR / "meta.json").open("w"), indent=2, ensure_ascii=False)

    imgs, sts, acts, boundaries = [], [], [], [0]
    for i, m in enumerate(metas):
        for sf in sorted((OUTPUT_DIR / f"ep_{i}").glob("step_*.npz")):
            d = np.load(sf)
            imgs.append(d["image"]); sts.append(d["state"]); acts.append(d["action"])
        boundaries.append(len(imgs))
    np.savez_compressed(OUTPUT_DIR / "consolidated.npz",
                        images=np.array(imgs, dtype=np.uint8),
                        states=np.array(sts, dtype=np.float32),
                        actions=np.array(acts, dtype=np.float32),
                        episode_boundaries=np.array(boundaries, dtype=np.int32))

    print(f"\n{'='*55}")
    print(f"  DONE — {meta['total_frames']} frames, {meta['total_episodes']} episodes")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"    meta.json         — dataset metadata")
    print(f"    consolidated.npz  — images+states+actions")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
