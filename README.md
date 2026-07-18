# LingBot-VLA Panda — PickPlaceBox 可用备份

> **备份日期**: 2026-07-18  
> **原始仓库**: https://github.com/Pleyxo/lingbot-vla-panda  
> **状态**: 训练完成 + 推理 Bug 已修复（Recording_1）

## 概述

Panda 机械臂 PickPlaceBox 任务的 VLA（Vision-Language-Action）仿真系统。  
任务：抓取红色方块并放入开口盒子。

- **机械臂**: Franka Panda (7-DOF)
- **仿真引擎**: MuJoCo + robosuite
- **GPU**: RTX 4090 24GB
- **基座模型**: Qwen2.5-VL-3B-Instruct + lingbot-vla-4b

## 文件结构

```
├── README.md
├── env_setup.sh              # 环境变量一键配置
├── test_env.py               # 环境验证脚本
├── vla_project/
│   ├── infer_panda.py        # 自主推理脚本（已修复 action key 问题）
│   ├── pick_place_box.py     # PickPlaceBox 自定义环境
│   ├── table_bin_arena.py    # 桌面+盒子场景
│   ├── reconstruct_dataset.py # 数据重建（离线回放）
│   ├── compute_norm.py       # 归一化统计量计算
│   ├── convert_to_lerobot.py # 转换为 LeRobot 格式
│   ├── patch_feature_type.py # 特征类型补丁
│   ├── train_panda.sh        # 一键训练脚本
│   └── train_panda.yaml      # 训练配置文件
├── configs/
│   └── robot_configs/
│       └── panda.yaml        # Panda 机器人配置（action/state 映射）
├── assets/
│   └── norm_stats/
│       └── panda.json        # 归一化统计量（14-dim state + 8-dim action）
├── web_teleop/
│   ├── server.py             # Web 遥操作服务器
│   └── replay_server.py      # 轨迹回放服务器
└── deploy/
    └── lingbot_vla_policy.py # VLA 策略推理接口
```

## 新服务器部署步骤

### 1. 系统依赖
```bash
sudo apt-get update
sudo apt-get install -y libegl1 libegl-mesa0 libegl1-mesa-dev
sudo apt-get install -y libosmesa6 libosmesa6-dev
sudo apt-get install -y libgl1-mesa-glx libgl1-mesa-dri libgles2
```

### 2. 克隆仓库
```bash
git clone https://github.com/Pleyxo/lingbot-vla-panda-useable1-.git
cd lingbot-vla-panda-useable1-
```

### 3. 克隆原始完整项目（需要其核心代码）
```bash
git clone https://github.com/Pleyxo/lingbot-vla-panda.git /data/coding/lingbot-vla
cd /data/coding/lingbot-vla
```

### 4. 创建 Conda 环境
```bash
conda env create -f environment.yml
conda activate lingbotvla
pip install -e .
pip install -e ./lingbotvla/models/vla/vision_models/lingbot-depth/ --no-deps
pip install -e ./lingbotvla/models/vla/vision_models/MoGe/
pip install https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz
```

### 5. 下载基座模型
```bash
mkdir -p /data/models
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /data/models/Qwen2.5-VL-3B-Instruct
huggingface-cli download robbyant/lingbot-vla-4b --local-dir /data/models/lingbot-vla-4b
```

### 6. 覆盖自定义文件（从本仓库）
```bash
cp -r vla_project/* /data/coding/lingbot-vla/vla_project/
cp configs/robot_configs/panda.yaml /data/coding/lingbot-vla/configs/robot_configs/
cp assets/norm_stats/panda.json /data/coding/lingbot-vla/assets/norm_stats/
cp web_teleop/*.py /data/coding/lingbot-vla/web_teleop/
cp deploy/lingbot_vla_policy.py /data/coding/lingbot-vla/deploy/
cp env_setup.sh test_env.py /data/coding/lingbot-vla/
```

### 7. 环境验证
```bash
source env_setup.sh
python test_env.py
# 应该显示 "ALL CHECKS PASSED"
```

## 工作流程

### Step 0: 录制遥操轨迹
```bash
cd web_teleop && python server.py
# 浏览器打开 http://<server>:9090
# 录制 5-10 条"抓取方块→放入盒子"的成功轨迹
```

### Step 1: 数据重建
```bash
python vla_project/reconstruct_dataset.py
```

### Step 2: 计算归一化
```bash
python vla_project/compute_norm.py
```

### Step 3: 训练
```bash
bash vla_project/train_panda.sh
```

### Step 4: 推理测试
```bash
MUJOCO_GL=osmesa python vla_project/infer_panda.py \
    --checkpoint vla_project/action_expert_lora/checkpoints/global_step_1000/hf_ckpt \
    --episodes 10
```

## 已知问题 & 修复记录

### Recording_1 (2026-07-18): 推理 steps=1 立即失败
- **问题**: 模型输出 `chunk["action"]` 联合键，但 infer_panda.py 查找 `chunk["action.arm.position"]` 导致 break
- **修复**: infer_panda.py 第 88-104 行，同时支持联合键和分离键两种格式
- **详见**: Panda_VLA_仿真系统问题排查和修改日志.docx

## 环境变量
```bash
export MUJOCO_GL=osmesa       # 无头渲染（Docker/服务器必须）
export XDG_RUNTIME_DIR=/tmp
export QWEN25_PATH=/data/models/Qwen2.5-VL-3B-Instruct
export PYTHONPATH=/data/coding/lingbot-vla:$PYTHONPATH
export HF_HUB_OFFLINE=1       # 离线模式
export TRANSFORMERS_OFFLINE=1
```
