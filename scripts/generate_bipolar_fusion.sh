#!/usr/bin/env bash
# Bipolar fusion. Default is contrast with alpha=2, the setting used in the later reports.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.env"
cd "${PROJECT_ROOT}"
PY="${PYTHON:-python}"
mkdir -p "${PROJECT_ROOT}/outputs/generate"
"${PY}" src/generation/generate_bipolar_fusion.py \
  --bits "${BITS:-1110}" \
  --length "${LENGTH:-18}" \
  --n "${N:-8}" \
  --kind "${KIND:-contrast}" \
  --alpha "${ALPHA:-2.0}" \
  --seed "${SEED:-42}"
