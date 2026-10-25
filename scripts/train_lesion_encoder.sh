#!/bin/bash
# Encoder-only training / ablation (evaluation/train_lesion_encoder.py).
# First run builds a ~1.3 GB crop cache in outputs/ (~11 min); later runs reuse it,
# and outputs/nodule_type.csv (type labels, data_prep/build_nodule_type.py) is built if missing.
#   bash scripts/train_lesion_encoder.sh                          # attn: none, ce, ce_mil
#     -> outputs/lesion_enc_attn_{none,ce,ce_mil}_seed0/best.pt (+ metrics.json)
#        logs outputs/seed_sweep_logs/lesion_enc_attn_{none,ce,ce_mil}_seed0.log
#   ARCHS="gap center attn" TYPE_LOSSES=none bash scripts/train_lesion_encoder.sh   # the pooling comparison
#   TYPE_LOSSES=ce_mil W_MIL=1.0 bash scripts/train_lesion_encoder.sh
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
export PYTHONWARNINGS=ignore
VENV="${VENV:-/root/notebooks/automl/env}"
PY="$VENV/bin/python"
DATA_ROOT="${DATA_ROOT:-/root/notebooks/groups/BME/reg2rg_nifti}"
ARCHS="${ARCHS:-attn}"
TYPE_LOSSES="${TYPE_LOSSES:-none ce ce_mil}"
EPOCHS="${EPOCHS:-40}"
SEED="${SEED:-0}"
W_TYPE="${W_TYPE:-1.0}"
W_MIL="${W_MIL:-0.5}"
MIL_WARMUP="${MIL_WARMUP:-10}"
mkdir -p outputs/seed_sweep_logs
for arch in $ARCHS; do
  for tl in $TYPE_LOSSES; do
    log="outputs/seed_sweep_logs/lesion_enc_${arch}_${tl}_seed${SEED}.log"
    LOGS="${LOGS:-} $log"
    echo "=== $arch type_loss=$tl -> $log"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" evaluation/train_lesion_encoder.py \
        --arch "$arch" --type_loss "$tl" --data_root "$DATA_ROOT" --epochs "$EPOCHS" --seed "$SEED" \
        --w_type "$W_TYPE" --w_mil "$W_MIL" --mil_warmup "$MIL_WARMUP" 2>&1 | tee "$log"
  done
done
echo
echo "=== summary"
grep -h "^BEST" $LOGS
