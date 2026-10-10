#!/usr/bin/env bash
# Retrain one of the five six-attribute finetune methods on the length expert.
# FINETUNE_MODE: lora_r16 | full | freeze_top | lora_r64 | bitfit
# Batch sizes are the ones used on the ~96GB training GPU. Override TRAIN_BS if needed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.env"
cd "${PROJECT_ROOT}"
export WANDB_MODE=disabled
PY="${PYTHON:-python}"

MODE="${FINETUNE_MODE:-lora_r16}"
CSV="${PROJECT_ROOT}/data/corpus/sequence_species_mic_binned_for_six_attr.csv"
OUT="${OUT_DIR:-${PROJECT_ROOT}/outputs/six_attr_finetune/${MODE}}"
mkdir -p "${OUT}"

FINETUNE_FLAG="${MODE}"
LORA_R=16
LORA_ALPHA=32
LR="1.5e-5"
TRAIN_BS_DEFAULT=128
GRAD_ACCUM_DEFAULT=2
GC=0
case "${MODE}" in
  lora_r16)
    FINETUNE_FLAG="lora"
    LORA_R=16
    LORA_ALPHA=32
    LR="1.5e-5"
    TRAIN_BS_DEFAULT=128
    GRAD_ACCUM_DEFAULT=2
    GC=0
    ;;
  full)
    LR="8e-6"
    TRAIN_BS_DEFAULT=168
    GRAD_ACCUM_DEFAULT=1
    GC=1
    ;;
  freeze_top|freeze_top2)
    FINETUNE_FLAG="freeze_top"
    LR="1e-5"
    TRAIN_BS_DEFAULT=168
    GRAD_ACCUM_DEFAULT=1
    GC=1
    ;;
  lora_r64)
    FINETUNE_FLAG="lora"
    LORA_R=64
    LORA_ALPHA=128
    LR="1.5e-5"
    TRAIN_BS_DEFAULT=78
    GRAD_ACCUM_DEFAULT=3
    GC=0
    ;;
  bitfit)
    LR="2e-4"
    TRAIN_BS_DEFAULT=90
    GRAD_ACCUM_DEFAULT=2
    GC=0
    ;;
  *)
    echo "unknown FINETUNE_MODE=${MODE}" >&2
    exit 1
    ;;
esac

TRAIN_BS="${TRAIN_BS:-${TRAIN_BS_DEFAULT}}"
GRAD_ACCUM="${GRAD_ACCUM:-${GRAD_ACCUM_DEFAULT}}"

ARGS=(
  --finetune_mode "${FINETUNE_FLAG}"
  --freeze_bottom_layers 10
  --gradient_checkpointing "${GC}"
  --bf16 1
  --csv_path "${CSV}"
  --init_model_dir "${PROJECT_ROOT}/models/AMPGuide"
  --tokenizer_dir "${PROJECT_ROOT}/models/AMPGuide"
  --output_dir "${OUT}"
  --overwrite_output_dir
  --train_only
  --max_length 256
  --pad_to_multiple_of 8
  --max_steps 5000
  --save_steps 500
  --save_total_limit 12
  --logging_steps 50
  --min_seq_len 1
  --train_bs "${TRAIN_BS}"
  --eval_bs 64
  --grad_accum "${GRAD_ACCUM}"
  --lr "${LR}"
  --dropout 0.08
  --label_smoothing 0.01
  --weight_decay 0.01
  --warmup_ratio 0.03
  --lr_scheduler cosine
  --num_workers 8
  --max_grad_norm 1.0
  --seed 42
  --use_staged_eval_schedule 0
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout 0.05
  --lora_target_modules c_attn,c_proj,c_fc
  --lora_modules_to_save lm_head
  --early_stop_patience 0
  --tf32
)
"${PY}" src/multi_attr/07.train_five_attr_finetune_modes.py "${ARGS[@]}"
