#!/bin/bash
# Probe the lesion token of a lesion-arm (B2) checkpoint for size and TW_Lung_RADS.
#
# Why: B2 seed 1 states nodule sizes that are uncorrelated with the reference
# (Spearman ~0.01 on val; every predicted size lands in 3-8 mm). This decides
# whether the token never encoded size (fix the encoder / add a loss) or encodes
# it and the decoder ignores it (fix the decoding side). See
# evaluation/extract_lesion_features.py for the full argument.
#
# Usage (from anywhere):
#   bash scripts/probe_lesion.sh                       # B2, seed 1, step 1390
#   SEED=2 STEP=1251 bash scripts/probe_lesion.sh
#   EXP=Reg2RG_ncku_lobe5_B3a ARCH=attn bash scripts/probe_lesion.sh
#
# Outputs: results/features_lesion_*.npz (git-ignored by results/features_*.npz),
#          outputs/seed_sweep_logs/probe_lesion_<exp>_seed<seed>_ckpt<step>.log
set -eu
export PYTHONWARNINGS=ignore   # silences sklearn worker warnings too (main-process filters do not reach joblib workers)

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

VENV="${VENV:-/root/notebooks/automl/env}"
PY="$VENV/bin/python"
DATA_ROOT="${DATA_ROOT:-/root/notebooks/groups/BME/reg2rg_nifti}"
EXP="${EXP:-Reg2RG_ncku_lobe5_lesion}"
SEED="${SEED:-1}"
STEP="${STEP:-1390}"
ARCH="${ARCH:-gap}"      # lesion encoder arch of this checkpoint: gap (B2) | attn | center

ckpt="outputs/${EXP}_seed${SEED}/checkpoint-${STEP}/pytorch_model.bin"
feat="results/features_lesion_${EXP}_seed${SEED}_ckpt${STEP}.npz"
rand_suffix=""; [ "$ARCH" != gap ] && rand_suffix="_$ARCH"
rand="results/features_lesion_random_init${rand_suffix}.npz"
log="outputs/seed_sweep_logs/probe_lesion_${EXP}_seed${SEED}_ckpt${STEP}.log"
mkdir -p results outputs/seed_sweep_logs

[ -f "$ckpt" ] || { echo "no checkpoint at $ckpt"; exit 1; }

# test split is deliberately not extracted
if [ ! -f "$feat" ]; then
    echo "[probe_lesion] extracting $feat"
    "$PY" evaluation/extract_lesion_features.py --ckpt "$ckpt" --lesion_arch "$ARCH" \
        --data_root "$DATA_ROOT" --splits train,val --out "$feat"
fi
if [ ! -f "$rand" ]; then
    echo "[probe_lesion] extracting random-init control $rand"
    "$PY" evaluation/extract_lesion_features.py --random_init --lesion_arch "$ARCH" \
        --data_root "$DATA_ROOT" --splits train,val --out "$rand"
fi

{
    echo "##### $(date -Is)  ckpt=$ckpt"
    echo "##### val only (clean: the encoder never trained on these patients)"
    "$PY" evaluation/probe_lesion.py --features model="$feat" rand="$rand" --splits val
    echo
    echo "##### train+val (tighter CIs; encoder has seen the train patients)"
    "$PY" evaluation/probe_lesion.py --features model="$feat" rand="$rand" --splits train,val
} 2>&1 | grep -v -i "warning" | tee "$log"
