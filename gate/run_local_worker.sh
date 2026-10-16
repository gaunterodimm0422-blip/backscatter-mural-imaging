#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_local_worker.sh <kind> <run_id>" >&2
  exit 2
fi

KIND="$1"
RUN_ID="$2"
: "${WORK_GATE:?WORK_GATE is required}"
: "${EXPORT_ROOT:?EXPORT_ROOT is required}"
: "${EXPORT_DIR:?EXPORT_DIR is required}"
: "${EXPORT_STAGING:?EXPORT_STAGING is required}"

cd "$WORK_GATE"
case "$KIND" in
  baseline)
    bash run_baseline.sh "$RUN_ID"
    ;;
  concrete_void)
    bash run_concrete_void.sh "$RUN_ID"
    ;;
  *)
    bash run_standard.sh "$KIND" "$RUN_ID"
    ;;
esac

if [[ "${SKIP_EXPORT:-0}" == "1" ]]; then
  echo "Local result ready: ${WORK_GATE}/output/${RUN_ID}"
  exit 0
fi

(
  flock 9
  cp -a "${WORK_GATE}/output/${RUN_ID}" "$EXPORT_STAGING"
  test -s "${EXPORT_STAGING}/hits.root"
  mv "$EXPORT_STAGING" "$EXPORT_DIR"
) 9>"${EXPORT_ROOT}/.backscatter_export.lock"
echo "Exported: ${EXPORT_DIR}"
