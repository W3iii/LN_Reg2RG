
# Inference on the NCKU ME test split (142 patients, 5 pulmonary lobes)

# ---- roots -------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="/root/notebooks/groups/BME/reg2rg_nifti"
CKPT_ROOT="$REPO_ROOT/pretrain"      # ~14.6 GB, git-ignored

experiment_name="Reg2RG_ncku_lobe5"

# Device settings
cuda_devices="0"

# Pretrained weights
lang_encoder_path="$CKPT_ROOT/Llama-2-7b-chat-hf"
tokenizer_path="$CKPT_ROOT/Llama-2-7b-chat-hf"
pretrained_visual_encoder="$CKPT_ROOT/Reg2RG/RadFM_vit3d.pth"
pretrained_adapter="$CKPT_ROOT/Reg2RG/RadFM_perceiver_fc.pth"

# Fine-tuned checkpoint produced by train_radgenome.sh
# TODO: point at the epoch you want to evaluate, e.g. .../checkpoint-2790/pytorch_model.bin
ckpt_path="$REPO_ROOT/outputs/$experiment_name/pytorch_model.bin"

# Data — switch report_file to region_report_val.csv for model selection
data_folder="$DATA_ROOT/images"
mask_folder="$DATA_ROOT/masks"
report_file="$DATA_ROOT/region_report_test.csv"

# Results — inside the project directory
result_path="$REPO_ROOT/results/ncku_lobe5_test_reports.csv"
