#!/bin/bash
# Cross-fitted per-nodule size / RADS predictions (evaluation/predict_lesion_attributes.py).
# First run builds a crop cache including test (~13 min); later runs reuse it.
#   bash scripts/predict_lesion_attributes.sh            # attn, seed 0
#   ARCH=center SEED=1 bash scripts/predict_lesion_attributes.sh
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PYTHONWARNINGS=ignore
VENV="${VENV:-/root/notebooks/automl/env}"
PY="$VENV/bin/python"
DATA_ROOT="${DATA_ROOT:-/root/notebooks/groups/BME/reg2rg_nifti}"
ARCH="${ARCH:-attn}"
SEED="${SEED:-0}"
mkdir -p outputs/seed_sweep_logs
log="outputs/seed_sweep_logs/lesion_pred_${ARCH}_seed${SEED}.log"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" evaluation/predict_lesion_attributes.py \
    --arch "$ARCH" --data_root "$DATA_ROOT" --seed "$SEED" 2>&1 | tee "$log"
