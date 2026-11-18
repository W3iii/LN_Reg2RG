
# Inference for split C + GT v3 (Impression, [TYPE#k]) -- see docs/V2_PIPELINE.md
#   split=val (choose the checkpoint) -> split=test_me and split=test_ldct (once)
#   experiment_name=Reg2RG_v3_vitq seed=42 ckpt_step=1360 split=test_ldct bash test_radgenome.sh ncku_a100_v3
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/notebooks/groups/BME/reg2rg_nifti}"
CKPT_ROOT="$REPO_ROOT/pretrain"

experiment_name="${experiment_name:-Reg2RG_v3_vitq}"
seed="${seed:-42}"
cuda_devices="${cuda_devices:-0}"

lang_encoder_path="$CKPT_ROOT/Llama-2-7b-chat-hf"
tokenizer_path="$CKPT_ROOT/Llama-2-7b-chat-hf"
pretrained_visual_encoder="$CKPT_ROOT/Reg2RG/RadFM_vit3d.pth"
pretrained_adapter="$CKPT_ROOT/Reg2RG/RadFM_perceiver_fc.pth"

# 136 steps per epoch -> epoch 10 = 1360; `ls outputs/<experiment_name>_seed<seed>` for the others
ckpt_step="${ckpt_step:-1360}"
run_dir="${experiment_name}${seed:+_seed$seed}"
ckpt_path="${ckpt_path:-$REPO_ROOT/outputs/$run_dir/checkpoint-${ckpt_step}/pytorch_model.bin}"
run_tag="${run_tag:-v3}"

split="${split:-val}"            # val | test_me | test_ldct
data_folder="$DATA_ROOT/images"
mask_folder="$DATA_ROOT/masks"
report_file="$DATA_ROOT/region_report_v3_${split}.csv"
monai_cache_dir="$DATA_ROOT/cache"

# Must match training (same encoder arch, lesion tokens on/off)
nodule_metadata="${nodule_metadata-$DATA_ROOT/nodule_metadata_v2.csv}"
lesion_arch="${lesion_arch:-vitq}"
lesion_pred_csv=""
lesion_order="truth"
lesion_size_unit="mm"
size_placeholder="False"
lesion_type="False"
lesion_type_min_conf="0.0"

# Output columns: GT_combined_report / Pred_combined_report filled with mask sizes,
# GT_raw_report / Pred_raw_report with [SIZE#k], Masked_regions = lobes to leave out of scoring
result_path="$REPO_ROOT/results/${run_dir}_${split}_ckpt${ckpt_step}_${run_tag}.csv"

# [TYPE#..] fill: per-nodule predictions of the same frozen encoder (evaluation/predict_types_v2.py);
# a type word is written only when its confidence reaches type_fill_min_conf, else the tag is removed.
# type_fill_csv= (empty) removes every tag -> the no-type-word baseline.
type_fill_csv="${type_fill_csv-$REPO_ROOT/outputs/type_pred_vitq_v2.csv}"
type_fill_min_conf="${type_fill_min_conf:-0.6}"
