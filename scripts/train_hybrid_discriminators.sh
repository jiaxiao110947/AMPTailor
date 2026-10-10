#!/usr/bin/env bash
# Retrain the five expert-backbone discriminators.
# GP/GN/FU/SF: amino-acid ids, LoRA r=32, select by validation AUROC.
# MIC: atomic SPxx token plus peptide, select by validation balanced accuracy.
# Published weights already live in models/discriminators. This writes to outputs/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.env"
cd "${PROJECT_ROOT}"
export WANDB_MODE=disabled
PY="${PYTHON:-python}"
OUT="${OUT_DIR:-${PROJECT_ROOT}/outputs/discriminators_hybrid}"
mkdir -p "${OUT}"
"${PY}" src/discriminators/train_six_attr_discriminators_hybrid.py \
  --csv_path "${PROJECT_ROOT}/data/corpus/sequence_species_mic_binned_for_six_attr.csv" \
  --backbone "${PROJECT_ROOT}/models/AMPGuide" \
  --split_root "${PROJECT_ROOT}/data/splits" \
  --tok_dir "${PROJECT_ROOT}/models/discriminators/mic_tokenizer" \
  --out_root "${OUT}" \
  --wandb_mode disabled
echo "[OK] hybrid discriminators finished: ${OUT}"
echo "Earlier expert-backbone recipes (v1, v2, v3, aa_vocab, fullft, lora_atomic) are in src/discriminators/."
