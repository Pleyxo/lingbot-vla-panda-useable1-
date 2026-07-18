#!/bin/bash
# LingBot-VLA Environment Setup for Single RTX 4090
# Usage: source env_setup.sh

export MUJOCO_GL=osmesa
export PYTHONPATH=/data/coding/lingbot-vla:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false

# Activate conda environment
source /data/miniconda/etc/profile.d/conda.sh
conda activate lingbotvla

echo "=== LingBot-VLA Environment ==="
echo "Conda env: lingbotvla"
echo "MUJOCO_GL: $MUJOCO_GL"
echo "Python: $(which python3)"
echo "=============================="
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export QWEN25_PATH=/data/models/Qwen2.5-VL-3B-Instruct
