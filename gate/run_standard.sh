#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_standard.sh <thin_layer|rod|double_rod|point|double_point|edge> <run_id>" >&2
  exit 2
fi

KIND="$1"
RUN_ID="$2"

case "$KIND" in
  thin_layer)
    export PHANTOM_MAC="phantom_thin_layer.mac"
    ;;
  rod)
    export PHANTOM_MAC="phantom_rod.mac"
    ;;
  double_rod)
    export PHANTOM_MAC="phantom_double_rod.mac"
    ;;
  point)
    export PHANTOM_MAC="phantom_point.mac"
    ;;
  double_point)
    export PHANTOM_MAC="phantom_double_point.mac"
    ;;
  edge)
    export PHANTOM_MAC="phantom_edge.mac"
    ;;
  *)
    echo "Unknown standard target kind: $KIND" >&2
    exit 2
    ;;
esac

export PHANTOM_ID="$KIND"
export PHANTOM_VOLUME="target"
export TRUTH_MAC="truth_target.mac"
export TRUTH_OUTPUT_SUFFIX="_truth_target"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_baseline.sh" "$RUN_ID"
