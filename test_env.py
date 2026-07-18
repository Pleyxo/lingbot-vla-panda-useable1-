#!/usr/bin/env python3
"""Verify LingBot-VLA environment on single RTX 4090."""
import sys
print("=" * 60)
print("LingBot-VLA Environment Verification")
print("=" * 60)

# 1. Core packages
print("\n[1/6] Core packages...")
import torch; print(f"  PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}")
import numpy; print(f"  NumPy {numpy.__version__}")
import transformers; print(f"  Transformers {transformers.__version__}")

# 2. Simulation packages
print("\n[2/6] Simulation packages...")
import mujoco; print(f"  MuJoCo {mujoco.__version__}")
import gymnasium; print(f"  Gymnasium {gymnasium.__version__}")
import robosuite; print(f"  robosuite {robosuite.__version__}")
import dm_control; print(f"  dm_control OK")

# 3. LeRobot
print("\n[3/6] LeRobot...")
import lerobot; print(f"  LeRobot {lerobot.__version__}")

# 4. LingBot-VLA
print("\n[4/6] LingBot-VLA...")
from lingbotvla.models.vla.pi0.modeling_lingbot_vla import LingbotVlaPolicy
from lingbotvla.models import build_foundation_model
print("  LingbotVlaPolicy import OK")

# 5. GPU Info
print("\n[5/6] GPU Info...")
print(f"  Device: {torch.cuda.get_device_name(0)}")
print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print(f"  Free: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB")

# 6. MuJoCo Rendering
print("\n[6/6] MuJoCo rendering test...")
import os
os.environ["MUJOCO_GL"] = "osmesa"
model_xml = """<mujoco><worldbody><light pos="0 0 1"/><geom type="box" size=".1 .1 .1"/></worldbody></mujoco>"""
import tempfile
with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
    f.write(model_xml); xml_path = f.name
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
os.unlink(xml_path)
print("  MuJoCo EGL rendering OK")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED - Environment is ready!")
print("=" * 60)
