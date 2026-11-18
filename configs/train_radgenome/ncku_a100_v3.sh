
# Split C + GT v3 (ME + LDCT): Findings per lobe + Impression, [SIZE#k] and [TYPE#k] -- docs/V2_PIPELINE.md
#   train = ME 1,093 patients, val = ME 157; test_me 150 and test_ldct 566 are inference only.
#
# Default arm: frozen vitq lesion encoder (3 tokens per nodule), pretrained with
#   ARCHS=vitq META=$DATA_ROOT/nodule_metadata_v2.csv bash scripts/train_lesion_encoder.sh
# Other arms, same seed, own experiment_name:
#   experiment_name=Reg2RG_v3_B0 nodule_metadata= bash train_radgenome.sh ncku_a100_v3      # no lesion tokens
#   experiment_name=Reg2RG_v3_attn lesion_arch=attn \
#     lesion_encoder_ckpt=$REPO_ROOT/outputs/lesion_enc_attn_ce_mil_v2_seed0/best.pt bash train_radgenome.sh ncku_a100_v3
experiment_name="${experiment_name:-Reg2RG_v3_vitq}"
bf16=True
seed="${seed:-42}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/root/notebooks/groups/BME/reg2rg_nifti}"
CKPT_ROOT="$REPO_ROOT/pretrain"

cuda_devices="${cuda_devices:-0}"
master_port=25368

lang_encoder_path="$CKPT_ROOT/Llama-2-7b-chat-hf"
tokenizer_path="$CKPT_ROOT/Llama-2-7b-chat-hf"
pretrained_visual_encoder="$CKPT_ROOT/Reg2RG/RadFM_vit3d.pth"
pretrained_adapter="$CKPT_ROOT/Reg2RG/RadFM_perceiver_fc.pth"

data_folder="$DATA_ROOT/images"
mask_folder="$DATA_ROOT/masks"
# v2 / v3 are detected from this CSV's nodule_ids / loss_mask columns:
#   lesion slots = the lobe's nodule_ids (mask size, largest first), prompt '#k <lesion..>',
#   targets carry [SIZE#k], loss_mask=0 lobes are excluded from the loss.
report_file="$DATA_ROOT/region_report_v3_train.csv"
monai_cache_dir="$DATA_ROOT/cache"

# ${var-default}: an explicitly empty value (nodule_metadata=) is kept -> lesion tokens off
nodule_metadata="${nodule_metadata-$DATA_ROOT/nodule_metadata_v2.csv}"
lesion_arch="${lesion_arch:-vitq}"
lesion_encoder_ckpt="${lesion_encoder_ckpt-$REPO_ROOT/outputs/lesion_enc_vitq_ce_mil_v2_seed0/best.pt}"
freeze_lesion_encoder="${freeze_lesion_encoder:-True}"

# v1-only options (B3c-B3g); the v2 dataset refuses them
lesion_pred_csv=""
lesion_order="truth"
lesion_size_unit="mm"
size_placeholder="False"
lesion_type="False"
lesion_type_min_conf="0.0"

output_dir="$REPO_ROOT/outputs/${experiment_name}_seed${seed}"
deepspeed_config=""

# 1,093 training patients, batch 1 x accumulation 8 -> 136 optimizer steps per epoch
learning_rate=5e-5
per_device_train_batch_size=1
num_train_epochs="${num_train_epochs:-10}"
gradient_accumulation_steps=8
evaluation_strategy="no"
save_strategy="epoch"
save_total_limit="${save_total_limit:-5}"
weight_decay=0.0
warmup_steps=20
lr_scheduler_type="constant_with_warmup"
dataloader_num_workers=8
logging_steps=1
