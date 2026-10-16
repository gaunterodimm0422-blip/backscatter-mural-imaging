#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_local_export.sh <baseline|concrete_void|thin_layer|rod|double_rod|point|double_point|edge> <run_id>" >&2
  exit 2
fi

KIND="$1"
RUN_ID="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-/home/vgate/backscatter_runs}"
EXPORT_ROOT="${EXPORT_ROOT:-${SCRIPT_DIR}/output}"
WORK_GATE="${LOCAL_ROOT}/${RUN_ID}/gate"
EXPORT_DIR="${EXPORT_ROOT}/${RUN_ID}"
EXPORT_STAGING="${EXPORT_ROOT}/.${RUN_ID}.partial"

if [[ -e "$WORK_GATE" ]]; then
  echo "Local run directory already exists: $WORK_GATE" >&2
  exit 2
fi
if [[ -e "$EXPORT_DIR" ]]; then
  echo "Export directory already exists: $EXPORT_DIR" >&2
  exit 2
fi
if [[ -e "$EXPORT_STAGING" ]]; then
  echo "Incomplete export staging directory already exists: $EXPORT_STAGING" >&2
  exit 2
fi

mkdir -p "${WORK_GATE}/materials" "$EXPORT_ROOT"
cp "${SCRIPT_DIR}"/*.mac "${WORK_GATE}/"
cp "${SCRIPT_DIR}"/*.sh "${WORK_GATE}/"
cp "${SCRIPT_DIR}/README.md" "${WORK_GATE}/"
cp "${SCRIPT_DIR}"/materials/*.db "${WORK_GATE}/materials/"

# From this point onward execute the VM-local copy.  A long-lived Bash process
# must not continue reading a VirtualBox shared file that may be updated while
# GATE is running.
exec env \
  WORK_GATE="$WORK_GATE" \
  EXPORT_ROOT="$EXPORT_ROOT" \
  EXPORT_DIR="$EXPORT_DIR" \
  EXPORT_STAGING="$EXPORT_STAGING" \
  SKIP_EXPORT="${SKIP_EXPORT:-0}" \
  bash "${WORK_GATE}/run_local_worker.sh" "$KIND" "$RUN_ID"
