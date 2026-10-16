#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_concrete_void.sh <run_id>" >&2
  exit 2
fi

RUN_ID="$1"
export PHANTOM_MAC="phantom_concrete_void.mac"
export PHANTOM_ID="${PHANTOM_ID:-concrete_100mm_cylindrical_void}"
export PHANTOM_VOLUME="concrete"
export TRUTH_MAC="truth_concrete.mac"
export TRUTH_OUTPUT_SUFFIX="_truth_concrete"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_baseline.sh" "$RUN_ID"
