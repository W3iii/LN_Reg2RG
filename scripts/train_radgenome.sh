#!/bin/bash

# 检查是否提供了参数
if [ $# -eq 0 ]
then
    echo "Warning: The configuration file should be specified according to the device being used!"
    exit 1
fi

# 获取配置文件名
config_file="$1"

# 获取当前脚本名称
script_name=$(basename "$0" .sh)

# 导入配置文件
source "../configs/${script_name}/${config_file}.sh"

# 根据 cuda_devices 设置的GPU数量自动设置 nproc_per_node
nproc_per_node=$(echo $cuda_devices | tr ',' '\n' | wc -l)

# DeepSpeed is optional: leave deepspeed_config empty in the config to skip it.
# With LoRA the optimizer state is small, so ZeRO-2 buys little on a single GPU,
# and deepspeed 0.12.6 does not import against torch >= 2.4
# (elastic_agent.py imports `log`, renamed to `logger` upstream).
ds_args=()
if [ -n "${deepspeed_config:-}" ]; then
    ds_args+=(--deepspeed "$deepspeed_config")
    echo "[train] DeepSpeed enabled: $deepspeed_config"
else
    echo "[train] DeepSpeed disabled (deepspeed_config is empty)"
fi

# Single GPU: launch with plain `python`, not torchrun.
#
# torchrun sets local_rank, so HF Trainer wraps the model in DDP even for one
# process. The model enables reentrant gradient checkpointing, whose backward
# replays the forward and fires autograd hooks twice per parameter; DDP then
# aborts with "Expected to mark a variable ready only once" on the first
# gradient sync (i.e. after gradient_accumulation_steps batches, so it looks
# like training started fine).
#
# Multi-GPU still needs torchrun. Expect the same conflict there — fix it at
# that point with non-reentrant checkpointing or DDP static graph, not by
# reverting this.
if [ "$nproc_per_node" -eq 1 ]; then
    echo "[train] single GPU -> plain python (no DDP)"
    launcher=(python)
else
    echo "[train] $nproc_per_node GPUs -> torchrun (DDP)"
    launcher=(torchrun --nproc_per_node="$nproc_per_node" --master-port="$master_port")
fi

# 使用配置参数执行训练
CUDA_VISIBLE_DEVICES=$cuda_devices "${launcher[@]}" ../src/${script_name}.py \
    "${ds_args[@]}" \
    --bf16 $bf16 \
    --lang_encoder_path "$lang_encoder_path" \
    --tokenizer_path "$tokenizer_path" \
    --pretrained_visual_encoder "$pretrained_visual_encoder" \
    --pretrained_adapter "$pretrained_adapter" \
    --data_folder "$data_folder" \
    --mask_folder "$mask_folder" \
    --report_file "$report_file" \
    --monai_cache_dir "$monai_cache_dir" \
    --nodule_metadata "$nodule_metadata" \
    --output_dir "$output_dir" \
    --seed $seed \
    --per_device_train_batch_size $per_device_train_batch_size \
    --num_train_epochs $num_train_epochs \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "$evaluation_strategy" \
    --save_strategy "$save_strategy" \
    --save_total_limit $save_total_limit \
    --learning_rate $learning_rate \
    --weight_decay $weight_decay \
    --warmup_steps $warmup_steps \
    --lr_scheduler_type "$lr_scheduler_type" \
    --dataloader_num_workers $dataloader_num_workers \
    --run_name "${experiment_name}_seed${seed}" \
    --logging_steps $logging_steps
