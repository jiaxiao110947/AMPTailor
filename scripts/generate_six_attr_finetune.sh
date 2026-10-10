#!/usr/bin/env bash
# Generate from one published six-attribute finetune checkpoint.
# METHOD: lora_r16 | full_ft | freeze_top2 | lora_r64 | bitfit
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.env"
cd "${PROJECT_ROOT}"
PY="${PYTHON:-python}"

METHOD="${METHOD:-lora_r16}"
MODEL="${PROJECT_ROOT}/models/six_attr_finetune/${METHOD}"
OUT="${OUT_DIR:-${PROJECT_ROOT}/outputs/generate/six_attr_finetune/${METHOD}}"
mkdir -p "${OUT}"

"${PY}" src/multi_attr/07.generate_five_attr_single_lora.py \
  --model_dir "${MODEL}" \
  --tokenizer_dir "${PROJECT_ROOT}/models/AMPGuide" \
  --out_fasta "${OUT}/regular6.fa" \
  --use_regular6 \
  --lengths "${LENGTHS:-18}" \
  --n_per_combo "${N_PER_COMBO:-8}" \
  --enforce_exact_length 1 \
  --temperature 1.0 \
  --top_p 0.95 \
  --repetition_penalty 1.05 \
  --seed "${SEED:-42}"
echo "wrote ${OUT}/regular6.fa"
