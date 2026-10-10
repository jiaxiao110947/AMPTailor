#!/usr/bin/env bash
# One LoRA adapter per attribute (gp, gn, fu, sf, len), initialized from the length expert.
# Published files are the final adapters in models/six_attr_disentangle.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.env"
cd "${PROJECT_ROOT}"
export WANDB_MODE=disabled
PY="${PYTHON:-python}"
OUT="${OUT_DIR:-${PROJECT_ROOT}/outputs/six_attr_disentangle}"
mkdir -p "${OUT}"
"${PY}" src/multi_attr/train_disentangled_attr_experts.py \
  --csv_path "${PROJECT_ROOT}/data/corpus/sequence_species_mic_binned_for_six_attr.csv" \
  --init_model_dir "${PROJECT_ROOT}/models/AMPGuide" \
  --tokenizer_dir "${PROJECT_ROOT}/models/AMPGuide" \
  --output_dir "${OUT}" \
  --max_steps 2500 \
  --train_bs 64 \
  --grad_accum 4 \
  --lr 1.5e-5 \
  --save_steps 500 \
  --seed 42
echo "[OK] disentangled adapters finished: ${OUT}"
