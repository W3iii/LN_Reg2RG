#!/bin/bash
# Inference + region diagnostic for every seed, to get the text metric's spread.
#
# The companion to probe_seeds.sh. All seeds are the same baseline arm, so the
# spread here is run-to-run noise on the metric docs/LESION_TOKENS.md §7 currently
# nominates as primary. Two equivalent runs have already produced 0.235 and 0.439
# micro-F1 while their training losses sat within 0.03 of each other, so this is
# measuring how much of that instability survives into the reported number.
#
# Result CSVs are tagged with the seed. test_radgenome.py appends and skips
# AccNums it finds already present, so sharing a filename across seeds would make
# the second and third runs silently produce nothing.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

VENV="${VENV:-/root/notebooks/automl/env}"
PY="$VENV/bin/python"
DATA_ROOT="${DATA_ROOT:-/root/notebooks/groups/BME/reg2rg_nifti}"
SPLIT="${SPLIT:-val}"
STEP="${STEP:-1390}"
seeds=("$@")
[ ${#seeds[@]} -eq 0 ] && seeds=(1 2 3)

mkdir -p outputs/seed_sweep_logs results

for s in "${seeds[@]}"; do
    ckpt="outputs/Reg2RG_ncku_lobe5_seed${s}/checkpoint-${STEP}/pytorch_model.bin"
    out="results/ncku_lobe5_${SPLIT}_seed${s}_ckpt${STEP}.csv"
    log="outputs/seed_sweep_logs/infer_seed${s}.log"

    if [ ! -f "$ckpt" ]; then
        echo "[infer] seed $s: no checkpoint, skipping"
        continue
    fi

    echo "[infer] === seed $s -> $out ($(date -Is)) ==="
    CUDA_VISIBLE_DEVICES=0 "$PY" src/test_radgenome.py \
        --lang_encoder_path pretrain/Llama-2-7b-chat-hf \
        --tokenizer_path pretrain/Llama-2-7b-chat-hf \
        --pretrained_visual_encoder pretrain/Reg2RG/RadFM_vit3d.pth \
        --pretrained_adapter pretrain/Reg2RG/RadFM_perceiver_fc.pth \
        --ckpt_path "$ckpt" \
        --data_folder "$DATA_ROOT/images" \
        --mask_folder "$DATA_ROOT/masks" \
        --report_file "$DATA_ROOT/region_report_${SPLIT}.csv" \
        --monai_cache_dir outputs/monai_cache \
        --nodule_metadata "" \
        --result_path "$out" > "$log" 2>&1

    if [ $? -ne 0 ]; then
        echo "[infer] !! seed $s failed, see $log"
        continue
    fi
    echo "[infer] seed $s done ($(date -Is))"

    "$PY" evaluation/analyze_region_predictions.py --result "$out" \
        > "outputs/seed_sweep_logs/diag_seed${s}.log" 2>&1 \
        || echo "[infer] seed $s diagnostic failed"
done

echo
echo "=== micro precision / recall / F1 across seeds ==="
for s in "${seeds[@]}"; do
    d="outputs/seed_sweep_logs/diag_seed${s}.log"
    [ -f "$d" ] || continue
    echo "  seed $s: $(grep -E '^micro' "$d" | head -1)"
done
echo "=== per-lobe recall (the headline for RML/RUL) ==="
for s in "${seeds[@]}"; do
    d="outputs/seed_sweep_logs/diag_seed${s}.log"
    [ -f "$d" ] || continue
    echo "  seed $s:"
    grep -E '^(RUL|RML|RLL|LUL|LLL) ' "$d" | tail -5 | sed 's/^/    /'
done
