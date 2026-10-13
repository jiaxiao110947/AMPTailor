#!/usr/bin/env bash
# Guided decoding: PeptidePrior + alpha * (AMPGuide - CounterAMPGuide).
# July 2026 production setting used alpha=2.0 and alpha_eos=0.0.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.env"
cd "${PROJECT_ROOT}"
PY="${PYTHON:-python}"
OUT="${OUT_DIR:-${PROJECT_ROOT}/outputs/generate/guided}"
mkdir -p "${OUT}"
"${PY}" src/generation/generate_amptailor_length_amp.py \
  --base_model_dir "${PROJECT_ROOT}/models/PeptidePrior" \
  --expert_model_dir "${PROJECT_ROOT}/models/AMPGuide" \
  --anti_model_dir "${PROJECT_ROOT}/models/CounterAMPGuide" \
  --tokenizer_dir "${PROJECT_ROOT}/models/PeptidePrior" \
  --out_dir "${OUT}" \
  --header_prefix "amptailor" \
  --target_length "${TARGET_LENGTH:-18}" \
  --nsamples "${NSAMPLES:-32}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
  --batch_size "${BATCH_SIZE:-32}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --top_k "${TOP_K:-0}" \
  --top_p "${TOP_P:-0.9}" \
  --alpha "${ALPHA:-2.0}" \
  --alpha_eos "${ALPHA_EOS:-0.0}" \
  --seed "${SEED:-42}" \
  --repetition_penalty "${GEN_REP_PENALTY:-1.18}" \
  --rep_window 64 \
  --max_homopolymer_run 6 \
  --no_repeat_ngram_size 4 \
  --keep_eos_after_filter
echo "[OK] guided FASTA written under ${OUT}"
