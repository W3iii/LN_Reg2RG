
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

# Which checkpoint to evaluate. trainer.save_state() at the end of training only
# writes optimizer/scheduler state — the weights live in checkpoint-<step>/, one
# per epoch (save_total_limit keeps the last few). `ls outputs/$experiment_name`
# to see what is available.
ckpt_step="1390"
ckpt_path="$REPO_ROOT/outputs/$experiment_name/checkpoint-${ckpt_step}/pytorch_model.bin"

# Distinguishes this run's result CSV from other runs that happen to land on the
# same checkpoint step number (e.g. a from-scratch retrain after a code change —
# same epoch math, same step numbers, but genuinely different weights). Bump this
# any time you retrain into the same experiment_name/output_dir.
run_tag="lesion_v1"

# Which split. Use val while choosing a checkpoint; switch to test only once.
split="val"

data_folder="$DATA_ROOT/images"
mask_folder="$DATA_ROOT/masks"
report_file="$DATA_ROOT/region_report_${split}.csv"

# Must match the checkpoint's training arm. Evaluating a lesion-trained model with
# this empty leaves every <lesion*> id pointing at a zero embedding, and nothing
# raises -- the reports just quietly get worse.
nodule_metadata="${nodule_metadata:-}"

# Results — tagged by split, checkpoint, and run_tag so runs do not overwrite (or
# get silently skipped against) each other. test_radgenome.py appends and skips
# AccNums already present, so reusing one filename across genuinely different
# models silently produces no new rows.
result_path="$REPO_ROOT/results/ncku_lobe5_${split}_ckpt${ckpt_step}_${run_tag}.csv"
