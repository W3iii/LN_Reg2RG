
# Experiment settings — NCKU ME dataset, 5 pulmonary lobes as regions
experiment_name="Reg2RG_ncku_lobe5"
bf16=True

# ---- roots -------------------------------------------------------------
# Dataset lives on the shared group volume; pretrained weights and everything
# this run produces stay inside the project directory.
# repo root, resolved from this config's own location so it works wherever you clone
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="/root/notebooks/groups/BME/reg2rg_nifti"
CKPT_ROOT="$REPO_ROOT/pretrain"      # ~14.6 GB, git-ignored

# Device settings — set to the A100 ids on your server
cuda_devices="0"

# Torchrun settings
master_port=25368

# Pretrained weights
lang_encoder_path="$CKPT_ROOT/Llama-2-7b-chat-hf"
tokenizer_path="$CKPT_ROOT/Llama-2-7b-chat-hf"
pretrained_visual_encoder="$CKPT_ROOT/Reg2RG/RadFM_vit3d.pth"
pretrained_adapter="$CKPT_ROOT/Reg2RG/RadFM_perceiver_fc.pth"

# Data — produced by DataSet/reg2rg_dataset/{export_reg2rg,make_csv_split}.py
data_folder="$DATA_ROOT/images"
mask_folder="$DATA_ROOT/masks"
report_file="$DATA_ROOT/region_report_train.csv"
monai_cache_dir="$DATA_ROOT/cache" # useless

# Outputs — inside the project directory
output_dir="$REPO_ROOT/outputs/$experiment_name"

# DeepSpeed — empty means "don't use it".
# On a single 80 GB A100 with LoRA, ZeRO-2 mainly shards optimizer state, which
# is already tiny here, so it is not worth the version friction: deepspeed
# 0.12.6 fails to import against torch >= 2.4. Set this to
# "$REPO_ROOT/ds_configs/stage2.json" only when going multi-GPU, and then pin a
# deepspeed release that matches your torch (>= 0.15 for torch >= 2.4).
deepspeed_config=""

# Training settings
# NOTE: only 1116 training patients (RadGenome has ~20k), so 10 epochs will very
# likely overfit. Checkpoints are saved every epoch — evaluate several, do not
# assume the last one is best.
learning_rate=5e-5
per_device_train_batch_size=1
num_train_epochs=10
gradient_accumulation_steps=8
evaluation_strategy="no"
save_strategy="epoch"
save_total_limit=3
weight_decay=0.0
warmup_steps=20
lr_scheduler_type="constant_with_warmup"
dataloader_num_workers=8
logging_steps=1
