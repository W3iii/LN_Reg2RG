#!/bin/bash
# Encoder-only training / ablation (evaluation/train_lesion_encoder.py).
# First run builds a ~1.3 GB crop cache in outputs/ (~11 min); later runs reuse it,
# and outputs/nodule_type.csv (type labels, data_prep/build_nodule_type.py) is built if missing.
#   bash scripts/train_lesion_encoder.sh                          # attn: none, ce, ce_mil
#     -> outputs/lesion_enc_attn_{none,ce,ce_mil}_seed0/best.pt (+ metrics.json)
#        logs outputs/seed_sweep_logs/lesion_enc_attn_{none,ce,ce_mil}_seed0.log
#   ARCHS="gap center attn" TYPE_LOSSES=none bash scripts/train_lesion_encoder.sh   # the pooling comparison
#   TYPE_LOSSES=ce_mil W_MIL=1.0 bash scripts/train_lesion_encoder.sh
#   TYPE_LOSSES=ce_mil CONCEPT=1 bash scripts/train_lesion_encoder.sh   # + concept bottleneck g(size, type) -> RADS
#     -> outputs/lesion_enc_attn_ce_mil_cbm_seed0/, log ..._ce_mil_cbm_seed0.log
#   TYPE_LOSSES=ce_mil CONCEPT=frozen bash scripts/train_lesion_encoder.sh   # arm C: g pretrained + frozen, grade loss -> type head
#     -> outputs/lesion_enc_attn_ce_mil_cbmfrozen_seed0/
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
CONCEPT="${CONCEPT:-0}"
W_CBM="${W_CBM:-1.0}"
W_CONSIST="${W_CONSIST:-0.1}"
case "$CONCEPT" in
  1|joint) CBM_ARGS="--concept joint --w_cbm $W_CBM --w_consist $W_CONSIST"; CBM_SFX=_cbm ;;
  frozen)  CBM_ARGS="--concept frozen --w_cbm ${W_CBM_FROZEN:-0.3}"; CBM_SFX=_cbmfrozen ;;
  *)       CBM_ARGS=""; CBM_SFX="" ;;
esac
mkdir -p outputs/seed_sweep_logs
for arch in $ARCHS; do
  for tl in $TYPE_LOSSES; do
    log="outputs/seed_sweep_logs/lesion_enc_${arch}_${tl}${CBM_SFX}_seed${SEED}.log"
    LOGS="${LOGS:-} $log"
    echo "=== $arch type_loss=$tl concept=$CONCEPT -> $log"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PY" evaluation/train_lesion_encoder.py \
        --arch "$arch" --type_loss "$tl" --data_root "$DATA_ROOT" --epochs "$EPOCHS" --seed "$SEED" \
        --w_type "$W_TYPE" --w_mil "$W_MIL" --mil_warmup "$MIL_WARMUP" $CBM_ARGS 2>&1 | tee "$log"
  done
done
echo
echo "=== summary"
grep -h "^BEST\|^LAST" $LOGS
