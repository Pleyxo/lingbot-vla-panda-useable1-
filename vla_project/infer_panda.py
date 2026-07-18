#!/usr/bin/env python3
"""
Panda VLA 自主推理脚本 — PickPlaceBox 任务
===========================================================================
加载微调后的 VLA 模型，在 robosuite PickPlaceBox 仿真中闭环控制机械臂。
模型通过 agentview 相机观察场景，自主完成"抓取红色方块并放入开口盒子"任务。

用法:
  python infer_panda.py --checkpoint /path/to/checkpoint --episodes 10
"""

import os, sys, time, argparse
import numpy as np

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"
os.environ["PYTHONPATH"] = "/data/coding/lingbot-vla:" + os.environ.get("PYTHONPATH", "")

import torch

# 导入自定义 PickPlaceBox 环境
sys.path.insert(0, "/data/coding/lingbot-vla/vla_project")
from pick_place_box import PickPlaceBox

sys.path.insert(0, "/data/coding/lingbot-vla")
from deploy.lingbot_vla_policy import LingbotVLAServer

# ── Same state layout as reconstruct_dataset.py ──
STATE_KEYS = [
    "robot0_joint_pos_cos",   # 7
    "robot0_eef_pos",         # 3
    "robot0_gripper_qpos",    # 1
    "cube_pos",               # 3
]  # total: 14 dim


def make_env():
    """Create PickPlaceBox environment: pick cube and place into open box."""
    return PickPlaceBox(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
        control_freq=20,
        horizon=500,
        reward_shaping=True,
    )


def build_obs(env, obs: dict, camera: str, img_size: int = 224) -> dict:
    """Build observation dict for LingbotVLAServer.infer()."""
    img = env.sim.render(width=img_size, height=img_size, camera_name=camera)
    img = np.flipud(img)

    # 14-dim state (gripper_qpos is (2,), take first)
    parts = []
    for k in STATE_KEYS:
        v = obs[k].flatten()
        if k == "robot0_gripper_qpos":
            v = v[:1]
        parts.append(v)
    state = np.concatenate(parts).astype(np.float32)

    return {
        "observation.images.agentview": img.astype(np.uint8),
        "observation.state": state,
        "robo_name": "panda",
        "reset": False,
        "task": "Pick up the red cube and place it into the open box",
        "task_index": 0,
    }


def run_episode(env, policy: LingbotVLAServer, camera: str,
                img_size: int, max_steps: int = 500) -> dict:
    obs = env.reset()
    policy.reset("panda")
    total_reward = 0.0

    done = False
    for step in range(max_steps):
        obs_dict = build_obs(env, obs, camera, img_size)

        try:
            chunk = policy.infer(obs_dict)
        except Exception as e:
            print(f"  [WARN] inference error step {step}: {e}")
        if chunk is None:
            break

        # Support both combined "action" key and split "action.arm.position"/"action.effector.position" keys
        if "action" in chunk:
            # Combined format: [chunk_len, 8] = arm[7] + gripper[1]
            arm0 = chunk["action"][0, :7]
            grip0 = chunk["action"][0, 7:8]
        elif "action.arm.position" in chunk and "action.effector.position" in chunk:
            # Split format from robot config
            arm0 = chunk["action.arm.position"][0]
            grip0 = chunk["action.effector.position"][0]
        else:
            break

        action = np.concatenate([arm0, grip0])
        action = np.clip(action, -1.0, 1.0)
        if step == 0:
            print(f"  [DEBUG] raw_arm={arm0} raw_grip={grip0} action={np.round(action, 3)}")
            print(f"  [DEBUG] state={np.round(obs_dict["observation.state"], 3)}")
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)

        if step % 50 == 0:
            eef = obs["robot0_eef_pos"]
            cube = obs["cube_pos"]
            print(f"  step {step:3d}: eef={[f'{x:+.3f}' for x in eef]}  "
                  f"cube_z={cube[2]:.3f}  r={reward:.3f}")

        if done:
            break

    return {"success": bool(done), "steps": step + 1, "total_reward": total_reward}


def main():
    p = argparse.ArgumentParser(description="Panda VLA autonomous inference - PickPlaceBox")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--camera", type=str, default="agentview")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--use_length", type=int, default=10)
    p.add_argument("--norm_path", type=str,
                   default="/data/coding/lingbot-vla/assets/norm_stats/panda.json")
    args = p.parse_args()

    print("=" * 55)
    print("  Panda VLA – PickPlaceBox Autonomous Inference")
    print("=" * 55)
    print(f"  Task:       Pick up cube and place into open box")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Episodes:   {args.episodes}")

    print("\n[1/3] Loading VLA policy ...")
    policy = LingbotVLAServer(
        path_to_pi_model=args.checkpoint, use_length=args.use_length,
        use_bf16=True, robot_norm_path=args.norm_path,
        num_denoising_step=10, use_compile=False)
    print("  Model ready.")

    print("\n[2/3] Creating PickPlaceBox simulation ...")
    env = make_env()
    print(f"  DOF={env.robots[0].dof}  |  cameras={list(env.sim.model.camera_names)}")

    print(f"\n[3/3] Running {args.episodes} episodes ...\n")
    results = []
    for ep in range(args.episodes):
        print(f"--- Episode {ep+1}/{args.episodes} ---")
        t0 = time.time()
        m = run_episode(env, policy, args.camera, args.img_size)
        dt = time.time() - t0
        results.append(m)
        print(f"  {'OK' if m['success'] else 'FAIL'}  |  "
              f"steps={m['steps']}  reward={m['total_reward']:.1f}  dt={dt:.1f}s")
    env.close()

    n_ok = sum(r["success"] for r in results)
    print(f"\n{'='*55}")
    print(f"  RESULTS: {n_ok}/{args.episodes} ({n_ok/args.episodes*100:.0f}%)")
    print(f"  Avg steps: {np.mean([r['steps'] for r in results]):.0f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
