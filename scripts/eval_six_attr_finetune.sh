#!/usr/bin/env bash
# Generate regular6 and length-conditioned sequences, then score regular6
# with the five expert-backbone discriminators.
# METHOD: lora_r16 | full_ft | freeze_top2 | lora_r64 | bitfit
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.env"
cd "${PROJECT_ROOT}"
PY="${PYTHON:-python}"

METHOD="${METHOD:-lora_r16}"
MODEL="${PROJECT_ROOT}/models/six_attr_finetune/${METHOD}"
OUT="${OUT_DIR:-${PROJECT_ROOT}/outputs/eval/six_attr_finetune/${METHOD}}"
mkdir -p "${OUT}"

"${PY}" src/generation/eval_six_attr_finetune.py \
  --model_dir "${MODEL}" \
  --tokenizer_dir "${PROJECT_ROOT}/models/AMPGuide" \
  --out_dir "${OUT}" \
  --lengths "${LENGTHS:-6,8,10,12,16,20,24,28,32,40}" \
  --n_per_combo "${N_PER_COMBO:-10}" \
  --n_per_length "${N_PER_LENGTH:-20}" \
  --seed "${SEED:-42}"
echo "report ${OUT}/eval_report.json"
