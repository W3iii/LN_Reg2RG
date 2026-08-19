#!/bin/bash
# Probe every seed's checkpoint, to get the spread of the probe metric.
#
# The point is a comparison of *metrics*, not of models. All three seeds are the
# same baseline arm, so any spread here is run-to-run noise. Text metrics on this
# setup have already swung 0.235 to 0.439 micro-F1 between two equivalent runs,
# while their training losses sit within 0.03 of each other -- so the instability
# lives downstream of the fit, in generation and parsing. If the probe's spread is
# visibly tighter, it belongs as the primary metric and the four ablation arms in
# docs/LESION_TOKENS.md §7 need far fewer seeds each to be readable.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

VENV="${VENV:-/root/notebooks/automl/env}"
PY="$VENV/bin/python"
DATA_ROOT="${DATA_ROOT:-/root/notebooks/groups/BME/reg2rg_nifti}"
SPLIT="${SPLIT:-val}"
STEP="${STEP:-1390}"
seeds=("$@")
[ ${#seeds[@]} -eq 0 ] && seeds=(1 2 3)

mkdir -p outputs/seed_sweep_logs

for s in "${seeds[@]}"; do
    ckpt="outputs/Reg2RG_ncku_lobe5_seed${s}/checkpoint-${STEP}/pytorch_model.bin"
    feat="results/features_${SPLIT}_seed${s}.npz"
    log="outputs/seed_sweep_logs/probe_seed${s}.log"

    if [ ! -f "$ckpt" ]; then
        echo "[probe] seed $s: no checkpoint at $ckpt, skipping" | tee -a "$log"
        continue
    fi

    echo "[probe] === seed $s ($(date -Is)) ==="
    if [ ! -f "$feat" ]; then
        "$PY" evaluation/extract_region_features.py \
            --ckpt "$ckpt" \
            --data_root "$DATA_ROOT" \
            --tokenizer_path pretrain/Llama-2-7b-chat-hf \
            --split "$SPLIT" \
            --out "$feat" > "$log" 2>&1 || { echo "[probe] seed $s extract failed"; continue; }
    else
        echo "[probe] seed $s: reusing $feat"
    fi

    "$PY" evaluation/probe_local_info.py \
        --features "$feat" \
        --nodule_metadata "$DATA_ROOT/nodule_metadata.csv" >> "$log" 2>&1 \
        || echo "[probe] seed $s probe failed"
    echo "[probe] seed $s done"
done

echo
echo "=== post_fc vs lobe_voxels across seeds ==="
for s in "${seeds[@]}"; do
    log="outputs/seed_sweep_logs/probe_seed${s}.log"
    [ -f "$log" ] || continue
    pf=$(grep -E "^post_fc" "$log" | awk '{print $2}')
    lv=$(grep -E "^lobe_voxels" "$log" | awk '{print $2}')
    dl=$(grep "post_fc - lobe_voxels" "$log" | sed 's/.*delta //')
    echo "  seed $s: post_fc=$pf  lobe_voxels=$lv  delta=$dl"
done
