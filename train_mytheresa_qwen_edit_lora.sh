#!/usr/bin/env bash
set -euo pipefail

TRAIN_PY="${TRAIN_PY:-/home/work/MMSearch/cwf/search/ENSEMBLE/train_mytheresa_qwen_edit_lora.py}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/work/MMSearch/cwf/miniconda/envs/search-train/bin/accelerate}"

JSONL="${JSONL:-/home/work/MMSearch/cwf/search/outfit/mytheresa_recommender_reason_zh.handwritten_1_9330.qc_clean.all_revised.jsonl}"
PRETRAINED_MODEL="${PRETRAINED_MODEL:-/home/work/MMSearch/cwf/search/model/Qwen-Image-Edit-2511}"
TEXT_ENCODER_MODEL="${TEXT_ENCODER_MODEL:-/home/work/MMSearch/cwf/search/model/mytheresa_stylist_qwen25vl_7b_merged}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/work/MMSearch/svg-shared-model-new-copy/ai-search/outfit/finetune}"

WORLD_SIZE_IS_SET=0
RANK_IS_SET=0
if [[ -n "${WORLD_SIZE+x}" ]]; then
  WORLD_SIZE_IS_SET=1
fi
if [[ -n "${RANK+x}" ]]; then
  RANK_IS_SET=1
fi

WORLD_SIZE="${WORLD_SIZE:-${NUM_PROCESSES:-1}}"
RANK="${RANK:-${MACHINE_RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-${MAIN_PROCESS_PORT:-29501}}"

if [[ -z "${NUM_MACHINES+x}" ]]; then
  if [[ "${WORLD_SIZE_IS_SET}" -eq 1 && "${RANK_IS_SET}" -eq 1 ]]; then
    NUM_MACHINES="${WORLD_SIZE}"
  else
    NUM_MACHINES=1
  fi
fi
NUM_PROCESSES="${NUM_PROCESSES:-${WORLD_SIZE}}"
MACHINE_RANK="${MACHINE_RANK:-${RANK}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-${MASTER_PORT}}"

CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" && "${NUM_PROCESSES}" -eq 1 ]]; then
  CUDA_VISIBLE_DEVICES_VALUE=0
fi

LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_LAYERS="${LORA_LAYERS:-attn.to_q,attn.to_k,attn.to_v,attn.to_out.0,attn.add_q_proj,attn.add_k_proj,attn.add_v_proj,attn.to_add_out,img_mlp.net.0.proj,img_mlp.net.2}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
REFERENCE_GLOBAL_BATCH="${REFERENCE_GLOBAL_BATCH:-16}"
if [[ -z "${GRADIENT_ACCUMULATION_STEPS+x}" ]]; then
  PER_STEP_BATCH=$((TRAIN_BATCH_SIZE * NUM_PROCESSES))
  GRADIENT_ACCUMULATION_STEPS=$(((REFERENCE_GLOBAL_BATCH + PER_STEP_BATCH - 1) / PER_STEP_BATCH))
  if [[ "${GRADIENT_ACCUMULATION_STEPS}" -lt 1 ]]; then
    GRADIENT_ACCUMULATION_STEPS=1
  fi
fi
EFFECTIVE_GLOBAL_BATCH=$((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_PROCESSES))
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1720}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"
CHECKPOINTING_MODE="${CHECKPOINTING_MODE:-resume}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"

TARGET_IMAGE_AREA="${TARGET_IMAGE_AREA:-1048576}"
CONDITION_SHORT_SIDE_MIN="${CONDITION_SHORT_SIDE_MIN:-512}"
CONDITION_SHORT_SIDE_MAX="${CONDITION_SHORT_SIDE_MAX:-960}"
TEXT_ENCODER_MIN_PIXELS="${TEXT_ENCODER_MIN_PIXELS:-1024}"
TEXT_ENCODER_MAX_PIXELS="${TEXT_ENCODER_MAX_PIXELS:-262144}"

MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
REPORT_TO="${REPORT_TO:-wandb}"
RUN_NAME="${RUN_NAME:-mytheresa_qwen_edit_lora_attn_add_imgmlp_r${LORA_RANK}_gbs${EFFECTIVE_GLOBAL_BATCH}_steps${MAX_TRAIN_STEPS}_lr${LEARNING_RATE}_cond${CONDITION_SHORT_SIDE_MIN}-${CONDITION_SHORT_SIDE_MAX}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
LOG_PATH="${LOG_PATH:-${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log}"
WANDB_PROJECT="${WANDB_PROJECT:-mytheresa-qwen-image-edit-lora}"
WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"

DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-ddp}"
FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-FULL_SHARD}"
FSDP_AUTO_WRAP_POLICY="${FSDP_AUTO_WRAP_POLICY:-TRANSFORMER_BASED_WRAP}"
FSDP_TRANSFORMER_CLS="${FSDP_TRANSFORMER_CLS:-QwenImageTransformerBlock}"
FSDP_BACKWARD_PREFETCH="${FSDP_BACKWARD_PREFETCH:-BACKWARD_PRE}"
FSDP_STATE_DICT_TYPE="${FSDP_STATE_DICT_TYPE:-SHARDED_STATE_DICT}"
FSDP_USE_ORIG_PARAMS="${FSDP_USE_ORIG_PARAMS:-true}"
FSDP_OFFLOAD_PARAMS="${FSDP_OFFLOAD_PARAMS:-false}"
FSDP_ACTIVATION_CHECKPOINTING="${FSDP_ACTIVATION_CHECKPOINTING:-false}"

DEEPSPEED_ZERO_STAGE="${DEEPSPEED_ZERO_STAGE:-3}"
DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE="${DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE:-none}"
DEEPSPEED_OFFLOAD_PARAM_DEVICE="${DEEPSPEED_OFFLOAD_PARAM_DEVICE:-none}"
DEEPSPEED_ZERO3_INIT_FLAG="${DEEPSPEED_ZERO3_INIT_FLAG:-true}"
DEEPSPEED_ZERO3_SAVE_16BIT_MODEL="${DEEPSPEED_ZERO3_SAVE_16BIT_MODEL:-false}"

mkdir -p "${OUTPUT_DIR}"

ACCELERATE_ARGS=(
  launch
  --num_processes "${NUM_PROCESSES}"
  --num_machines "${NUM_MACHINES}"
  --mixed_precision "${MIXED_PRECISION}"
  --dynamo_backend no
  --main_process_port "${MAIN_PROCESS_PORT}"
)

case "${DISTRIBUTED_BACKEND}" in
  ddp)
    if [[ "${NUM_PROCESSES}" -gt 1 || "${NUM_MACHINES}" -gt 1 ]]; then
      ACCELERATE_ARGS+=(--multi_gpu)
    fi
    ;;
  fsdp)
    ACCELERATE_ARGS+=(
      --use_fsdp
      --fsdp_sharding_strategy "${FSDP_SHARDING_STRATEGY}"
      --fsdp_auto_wrap_policy "${FSDP_AUTO_WRAP_POLICY}"
      --fsdp_transformer_layer_cls_to_wrap "${FSDP_TRANSFORMER_CLS}"
      --fsdp_backward_prefetch "${FSDP_BACKWARD_PREFETCH}"
      --fsdp_state_dict_type "${FSDP_STATE_DICT_TYPE}"
      --fsdp_use_orig_params "${FSDP_USE_ORIG_PARAMS}"
      --fsdp_offload_params "${FSDP_OFFLOAD_PARAMS}"
      --fsdp_activation_checkpointing "${FSDP_ACTIVATION_CHECKPOINTING}"
    )
    ;;
  deepspeed)
    ACCELERATE_ARGS+=(
      --use_deepspeed
      --zero_stage "${DEEPSPEED_ZERO_STAGE}"
      --offload_optimizer_device "${DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE}"
      --offload_param_device "${DEEPSPEED_OFFLOAD_PARAM_DEVICE}"
      --zero3_init_flag "${DEEPSPEED_ZERO3_INIT_FLAG}"
      --zero3_save_16bit_model "${DEEPSPEED_ZERO3_SAVE_16BIT_MODEL}"
    )
    ;;
  *)
    echo "Unsupported DISTRIBUTED_BACKEND=${DISTRIBUTED_BACKEND}; expected ddp, fsdp, or deepspeed" >&2
    exit 1
    ;;
esac

if [[ "${NUM_MACHINES}" -gt 1 ]]; then
  ACCELERATE_ARGS+=(--machine_rank "${MACHINE_RANK}" --main_process_ip "${MASTER_ADDR}")
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "world_size=${WORLD_SIZE}, rank=${RANK}, num_processes=${NUM_PROCESSES}, num_machines=${NUM_MACHINES}, machine_rank=${MACHINE_RANK}"
echo "master_addr=${MASTER_ADDR}, main_process_port=${MAIN_PROCESS_PORT}, mixed_precision=${MIXED_PRECISION}"
echo "distributed_backend=${DISTRIBUTED_BACKEND}"
if [[ "${DISTRIBUTED_BACKEND}" == "fsdp" ]]; then
  echo "fsdp_sharding_strategy=${FSDP_SHARDING_STRATEGY}, fsdp_transformer_cls=${FSDP_TRANSFORMER_CLS}, fsdp_use_orig_params=${FSDP_USE_ORIG_PARAMS}, fsdp_offload_params=${FSDP_OFFLOAD_PARAMS}"
fi
if [[ "${DISTRIBUTED_BACKEND}" == "deepspeed" ]]; then
  echo "deepspeed_zero_stage=${DEEPSPEED_ZERO_STAGE}, offload_optimizer=${DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE}, offload_param=${DEEPSPEED_OFFLOAD_PARAM_DEVICE}"
fi
echo "train_batch_size=${TRAIN_BATCH_SIZE}, grad_accum=${GRADIENT_ACCUMULATION_STEPS}, effective_global_batch=${EFFECTIVE_GLOBAL_BATCH}, reference_global_batch=${REFERENCE_GLOBAL_BATCH}"
echo "max_train_steps=${MAX_TRAIN_STEPS}, checkpointing_steps=${CHECKPOINTING_STEPS}, checkpointing_mode=${CHECKPOINTING_MODE}"
echo "lora_rank=${LORA_RANK}, lora_alpha=${LORA_ALPHA}, learning_rate=${LEARNING_RATE}"
echo "lora_layers=${LORA_LAYERS}"
echo "target_image_area=${TARGET_IMAGE_AREA}, condition_short_side=${CONDITION_SHORT_SIDE_MIN}-${CONDITION_SHORT_SIDE_MAX}"
echo "report_to=${REPORT_TO}, wandb_project=${WANDB_PROJECT}, wandb_name=${WANDB_NAME}"
echo "output_dir=${OUTPUT_DIR}"
echo "log_path=${LOG_PATH}"
echo "accelerate_bin=${ACCELERATE_BIN}"
echo "accelerate_args=${ACCELERATE_ARGS[*]}"

TRAIN_ARGS=(
  "${TRAIN_PY}"
  --jsonl "${JSONL}"
  --pretrained-model "${PRETRAINED_MODEL}"
  --text-encoder-model "${TEXT_ENCODER_MODEL}"
  --output-dir "${OUTPUT_DIR}"
  --target-image-area "${TARGET_IMAGE_AREA}"
  --condition-short-side-min "${CONDITION_SHORT_SIDE_MIN}"
  --condition-short-side-max "${CONDITION_SHORT_SIDE_MAX}"
  --text-encoder-min-pixels "${TEXT_ENCODER_MIN_PIXELS}"
  --text-encoder-max-pixels "${TEXT_ENCODER_MAX_PIXELS}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --lr-warmup-steps "${LR_WARMUP_STEPS}"
  --max-train-steps "${MAX_TRAIN_STEPS}"
  --checkpointing-steps "${CHECKPOINTING_STEPS}"
  --checkpointing-mode "${CHECKPOINTING_MODE}"
  --rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-layers "${LORA_LAYERS}"
  --mixed-precision "${MIXED_PRECISION}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
  --report-to "${REPORT_TO}"
  --allow-tf32
  --gradient-checkpointing
)
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

env \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}" \
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_NAME="${WANDB_NAME}" \
  "${ACCELERATE_BIN}" "${ACCELERATE_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  "$@" 2>&1 | tee "${LOG_PATH}"
