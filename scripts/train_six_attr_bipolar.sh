#!/usr/bin/env bash
# Bipolar six-attribute LoRA on the length expert.
# Published weights: checkpoint-5000, train_only, no mid-training eval.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.env"
cd "${PROJECT_ROOT}"
export WANDB_MODE=disabled
PY="${PYTHON:-python}"

CSV="${PROJECT_ROOT}/data/corpus/sequence_species_mic_binned_for_six_attr.csv"
PREP="${PROJECT_ROOT}/outputs/six_attr_bipolar/prepared_data"
OUT="${OUT_DIR:-${PROJECT_ROOT}/outputs/six_attr_bipolar/run}"
mkdir -p "${PREP}" "${OUT}"

"${PY}" src/multi_attr/07.prepare_six_attr_bipolar_data.py \
  --csv_path "${CSV}" \
  --out_dir "${PREP}" \
  --min_seq_len 1 \
  --all_train

"${PY}" src/multi_attr/07.train_six_attr_bipolar_lora.py \
  --csv_path "${PREP}/amp_five_attr_source.csv" \
  --init_model_dir "${PROJECT_ROOT}/models/AMPGuide" \
  --tokenizer_dir "${PROJECT_ROOT}/models/AMPGuide" \
  --output_dir "${OUT}" \
  --overwrite_output_dir \
  --max_length 256 \
  --pad_to_multiple_of 8 \
  --max_steps 5000 \
  --eval_steps 500 \
  --save_steps 500 \
  --save_total_limit 12 \
  --logging_steps 50 \
  --eval_max_samples 3000 \
  --min_seq_len 1 \
  --train_bs 128 \
  --eval_bs 96 \
  --grad_accum 2 \
  --lr 1.5e-5 \
  --dropout 0.08 \
  --label_smoothing 0.01 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler cosine \
  --num_workers 8 \
  --max_grad_norm 1.0 \
  --seed 42 \
  --early_stop_patience 0 \
  --use_staged_eval_schedule 0 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules c_attn,c_proj,c_fc \
  --lora_modules_to_save lm_head \
  --tf32 \
  --train_only
echo "[OK] bipolar training finished: ${OUT}"
