#!/usr/bin/env bash
# Score a FASTA with GP / GN / FU / SF / MIC. There is no AMP head in this release.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/configs/paths.env"
cd "${PROJECT_ROOT}"
PY="${PYTHON:-python}"
if [[ $# -lt 1 ]]; then
  echo "usage: scripts/score_sequences.sh FASTA" >&2
  exit 1
fi
"${PY}" src/generation/score_expert_discriminators.py --fasta "$1"
