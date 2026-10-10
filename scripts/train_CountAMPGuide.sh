#!/usr/bin/env bash
# CountAMPGuide. Init: PeptidePrior. Corpus: nonAMP.txt, length 2-60.
# Published best_model is step 12000.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.env"
cd "${PROJECT_ROOT}"
export WANDB_MODE=disabled
PY="${PYTHON:-python}"

RAW="${PROJECT_ROOT}/data/corpus/nonAMP.txt"
DATA="${PROJECT_ROOT}/outputs/prepared/nonamp_length_train.txt"
STATS="${DATA}.stats.json"
INDEX="${PROJECT_ROOT}/outputs/prepared/CountAMPGuide_indexed"
OUT="${OUT_DIR:-${PROJECT_ROOT}/outputs/CountAMPGuide}"
mkdir -p "$(dirname "${DATA}")" "${OUT}"

if [[ ! -f "${DATA}" ]]; then
  "${PY}" src/pretrain/prepare_length_train_data.py \
    --input "${RAW}" \
    --output "${DATA}" \
    --stats_json "${STATS}" \
    --len_min 2 \
    --len_max 60 \
    --ensure_eos
fi
if [[ ! -f "${INDEX}/meta.json" || "${DATA}" -nt "${INDEX}/meta.json" ]]; then
  "${PY}" src/pretrain/prepare_indexed_splits.py \
    --data_file "${DATA}" \
    --out_dir "${INDEX}" \
    --seed 42 \
    --val_ratio 0.001 \
    --test_ratio 0.001 \
    --eval_max_samples 8000 \
    --ensure_eos
fi

export LENGTH_EVAL_ENABLE=1
export LENGTH_EVAL_LENGTHS="6,8,10,12,16,20,24,28,32,40"
export LENGTH_EVAL_N_PER_LEN=20
"${PY}" src/pretrain/train_model.py \
  --data_file "${DATA}" \
  --indexed_splits_dir "${INDEX}" \
  --base_model_dir "${PROJECT_ROOT}/models/PeptidePrior" \
  --tokenizer_dir "${PROJECT_ROOT}/models/BasePLM/gpt2_tokenizer" \
  --output_dir "${OUT}" \
  --max_length 128 \
  --max_steps 0 \
  --train_bs 1024 \
  --eval_bs 1024 \
  --grad_accum 1 \
  --num_workers 2 \
  --eval_max_samples 8000 \
  --lr 5e-5 \
  --warmup_ratio 0.02 \
  --lr_scheduler constant \
  --optim adamw_torch_fused \
  --logging_steps 20 \
  --eval_steps 4000 \
  --save_steps 4000 \
  --save_total_limit 10 \
  --load_best_model_at_end \
  --early_stop_patience 6 \
  --early_stop_min_delta 1e-4 \
  --eval_loss_trend_window 6 \
  --tf32 \
  --ensure_eos \
  --length_eval_enable
echo "[OK] anti training finished: ${OUT}"
