#!/bin/bash
# Cross-fitted per-nodule size / RADS / type predictions (evaluation/predict_lesion_attributes.py).
# First run builds a crop cache including test (~13 min); later runs reuse it.
#   bash scripts/predict_lesion_attributes.sh                     # attn, ce_mil -> lesion_pred_attn_ce_mil_seed0.csv
#     -> outputs/lesion_pred_attn_ce_mil_seed0.csv + _full.pt (B3f lesion_pred_csv / lesion_encoder_ckpt)
#   TYPE_LOSS=none bash scripts/predict_lesion_attributes.sh      # -> lesion_pred_attn_none_seed0.csv (B3e file untouched)
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PYTHONWARNINGS=ignore
VENV="${VENV:-/root/notebooks/automl/env}"
PY="$VENV/bin/python"
DATA_ROOT="${DATA_ROOT:-/root/notebooks/groups/BME/reg2rg_nifti}"
ARCH="${ARCH:-attn}"
TYPE_LOSS="${TYPE_LOSS:-ce_mil}"
SEED="${SEED:-0}"
W_TYPE="${W_TYPE:-1.0}"
W_MIL="${W_MIL:-0.5}"
MIL_WARMUP="${MIL_WARMUP:-10}"
tag="${ARCH}_${TYPE_LOSS}"
mkdir -p outputs/seed_sweep_logs
log="outputs/seed_sweep_logs/lesion_pred_${tag}_seed${SEED}.log"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" evaluation/predict_lesion_attributes.py \
    --arch "$ARCH" --type_loss "$TYPE_LOSS" --data_root "$DATA_ROOT" --seed "$SEED" \
    --w_type "$W_TYPE" --w_mil "$W_MIL" --mil_warmup "$MIL_WARMUP" 2>&1 | tee "$log"
