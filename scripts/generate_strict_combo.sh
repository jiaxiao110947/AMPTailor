#!/usr/bin/env bash
# Strict-combo joint prompt. bits=1110 means GP1 GN1 FU1 SF0.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.env"
cd "${PROJECT_ROOT}"
PY="${PYTHON:-python}"
mkdir -p "${PROJECT_ROOT}/outputs/generate"
"${PY}" src/generation/generate_strict_combo.py \
  --bits "${BITS:-1110}" \
  --length "${LENGTH:-18}" \
  --n "${N:-8}" \
  --seed "${SEED:-42}"
