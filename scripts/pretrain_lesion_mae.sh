#!/bin/bash
# MAE pretraining of the vitq lesion encoder on LIDC crops (evaluation/pretrain_lesion_mae.py),
# then the usual ME size / type / RADS training started from it (train_lesion_encoder.py --init_encoder).
#   crops: data_prep/lidc/build_lidc_mae.py on the workstation -> upload lidc_mae_crops.npy + .npy.csv to outputs/
#   bash scripts/pretrain_lesion_mae.sh                         # -> outputs/lesion_mae_vitq_lidc/mae.pt (+ recon_qc.png)
#   ARCHS=vitq TYPE_LOSSES=ce_mil META=$DATA_ROOT/nodule_metadata_v2.csv INIT=outputs/lesion_mae_vitq_lidc/mae.pt \
#     SEED=0 bash scripts/train_lesion_encoder.sh              # -> outputs/lesion_enc_vitq_ce_mil_v2_mae_seed0/best.pt
# Compare with the same command without INIT (outputs/lesion_enc_vitq_ce_mil_v2_seed<k>), several SEEDs each.
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PYTHONWARNINGS=ignore
VENV="${VENV:-/root/notebooks/automl/env}"
PY="$VENV/bin/python"
DATA="${DATA:-outputs/lidc_mae_crops.npy}"      # several files: DATA="a.npy b.npy"
OUT="${OUT:-outputs/lesion_mae_vitq_lidc}"
EPOCHS="${EPOCHS:-100}"
BS="${BS:-128}"
LR="${LR:-5e-4}"
MASK_RATIO="${MASK_RATIO:-0.6}"
mkdir -p outputs/seed_sweep_logs
log="outputs/seed_sweep_logs/$(basename "$OUT").log"
echo "=== MAE vitq on $DATA -> $OUT ($log)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" evaluation/pretrain_lesion_mae.py \
    --data $DATA --out "$OUT" --epochs "$EPOCHS" --bs "$BS" --lr "$LR" --mask_ratio "$MASK_RATIO" \
    ${ME_CACHE:+--me_cache $ME_CACHE} 2>&1 | tee "$log"
