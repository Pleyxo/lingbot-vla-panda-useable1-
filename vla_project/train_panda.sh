#!/bin/bash
# =============================================================================
# Launch Action-Expert-Only fine-tuning for Panda on single RTX 4090
# =============================================================================
set -e

export MUJOCO_GL=osmesa
export PYTHONPATH=/data/coding/lingbot-vla:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
source /data/miniconda/etc/profile.d/conda.sh
conda activate lingbotvla
cd /data/coding/lingbot-vla

echo "=============================================="
echo "  Panda VLA – Action Expert Fine-tuning"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "=============================================="

# ── Step 1: Reconstruct dataset ──
echo ""
echo "[Step 1/4] Reconstructing dataset from teleop data ..."
python /data/coding/lingbot-vla/vla_project/reconstruct_dataset.py \
    --camera agentview --img_size 224

# ── Step 2: Compute norm stats ──
echo ""
echo "[Step 2/4] Computing normalization statistics ..."
python /data/coding/lingbot-vla/vla_project/compute_norm.py \
    --data_dir /data/coding/lingbot-vla/vla_project/reconstructed_data \
    --output /data/coding/lingbot-vla/assets/norm_stats/panda.json

# ── Step 3: Train action expert ──
echo ""
echo "[Step 3/4] Starting training ..."
python tasks/vla/train_lingbotvla.py \
    --config-path /data/coding/lingbot-vla/vla_project/train_panda.yaml

echo ""
echo "[Step 4/4] Done! Checkpoints saved to vla_project/action_expert_lora/"
echo "=============================================="
