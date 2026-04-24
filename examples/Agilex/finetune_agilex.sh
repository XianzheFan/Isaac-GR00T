#!/usr/bin/env bash
set -x -euo pipefail

export NUM_GPUS="${NUM_GPUS:-1}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MAX_STEPS="${MAX_STEPS:-10000}"
USE_WANDB="${USE_WANDB:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"

DATASET_PATH="${DATASET_PATH:-/lustre/fs12/portfolios/llmservice/projects/llmservice_fm_vision/users/zhiqil/workspace/fxz/openpi/data/pnp_cup_0415}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/agilex_pnp_cup_0415_finetune}"

WANDB_FLAG=()
if [ "$USE_WANDB" = "1" ]; then
    WANDB_FLAG+=(--use_wandb)
fi

MASTER_PORT="${MASTER_PORT:-29500}"
if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCHER=(torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT")
else
    LAUNCHER=(python)
fi

"${LAUNCHER[@]}" \
    gr00t/experiment/launch_finetune.py \
    --base_model_path nvidia/GR00T-N1.7-3B \
    --dataset_path "$DATASET_PATH" \
    --modality_config_path examples/Agilex/agilex_config.py \
    --embodiment_tag NEW_EMBODIMENT \
    --num_gpus "$NUM_GPUS" \
    --output_dir "$OUTPUT_DIR" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 5 \
    --max_steps "$MAX_STEPS" \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    "${WANDB_FLAG[@]}" \
    --global_batch_size "$GLOBAL_BATCH_SIZE" \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --shard_size "$SHARD_SIZE" \
    --num_shards_per_epoch "$NUM_SHARDS_PER_EPOCH" \
    --episode_sampling_rate "$EPISODE_SAMPLING_RATE"
