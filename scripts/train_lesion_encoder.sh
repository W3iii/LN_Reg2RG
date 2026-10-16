#!/bin/bash
# Encoder-only comparison of lesion poolings (evaluation/train_lesion_encoder.py).
# First run builds a ~1.3 GB crop cache in outputs/ (~11 min); later runs reuse it.
#   bash scripts/train_lesion_encoder.sh             # gap, center, attn in sequence
#   ARCHS="center" EPOCHS=60 bash scripts/train_lesion_encoder.sh
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PYTHONWARNINGS=ignore
VENV="${VENV:-/root/notebooks/automl/env}"
PY="$VENV/bin/python"
DATA_ROOT="${DATA_ROOT:-/root/notebooks/groups/BME/reg2rg_nifti}"
ARCHS="${ARCHS:-gap center attn}"
EPOCHS="${EPOCHS:-40}"
SEED="${SEED:-0}"
mkdir -p outputs/seed_sweep_logs
for arch in $ARCHS; do
    log="outputs/seed_sweep_logs/lesion_enc_${arch}_seed${SEED}.log"
    echo "=== $arch -> $log"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" evaluation/train_lesion_encoder.py \
        --arch "$arch" --data_root "$DATA_ROOT" --epochs "$EPOCHS" --seed "$SEED" 2>&1 | tee "$log"
done
echo
echo "=== summary"
grep -h "^BEST" outputs/seed_sweep_logs/lesion_enc_*_seed${SEED}.log
