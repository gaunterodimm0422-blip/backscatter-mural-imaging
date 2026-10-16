#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash run_campaign_batch.sh <rods|double_rods_a|double_rods_b|edge|materials|edge_material|void_pilot>" >&2
  exit 2
fi

BATCH="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTROLLER_ROOT="${CONTROLLER_ROOT:-/home/vgate/backscatter_controller}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/vgate/backscatter_runs}"
RUN_TAG="${RUN_TAG:-}"
RUN_SUFFIX=""
ROD_PRIMARIES="${ROD_PRIMARIES:-5000000}"
DOUBLE_ROD_PRIMARIES="${DOUBLE_ROD_PRIMARIES:-5000000}"
EDGE_PRIMARIES="${EDGE_PRIMARIES:-2000000}"
MATERIAL_PRIMARIES="${MATERIAL_PRIMARIES:-2000000}"
VOID_PRIMARIES="${VOID_PRIMARIES:-2000000}"

if [[ -n "$RUN_TAG" ]]; then
  if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_TAG must contain only letters, digits, dot, underscore, or hyphen." >&2
    exit 2
  fi
  RUN_SUFFIX="_${RUN_TAG}"
fi
for variable_name in \
  ROD_PRIMARIES \
  DOUBLE_ROD_PRIMARIES \
  EDGE_PRIMARIES \
  MATERIAL_PRIMARIES \
  VOID_PRIMARIES
do
  variable_value="${!variable_name}"
  if [[ ! "$variable_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "${variable_name} must be a positive integer." >&2
    exit 2
  fi
done

mkdir -p "$CONTROLLER_ROOT"
cd "$SCRIPT_DIR"

declare -a PIDS=()
declare -a LABELS=()

launch() {
  local label="$1"
  shift
  local stdout_log="${CONTROLLER_ROOT}/${label}.stdout.log"
  local stderr_log="${CONTROLLER_ROOT}/${label}.stderr.log"
  local local_run_dir="${LOCAL_ROOT}/${label}/gate"
  if [[ -e "$local_run_dir" || -e "$stdout_log" || -e "$stderr_log" ]]; then
    echo "Refusing to overwrite an existing run or controller log: ${label}" >&2
    echo "Choose a new RUN_TAG; do not reuse an interrupted ROOT directory." >&2
    return 2
  fi
  echo "Launching ${label}"
  (
    env LOCAL_ROOT="$LOCAL_ROOT" SKIP_EXPORT=1 "$@"
  ) >"$stdout_log" 2>"$stderr_log" &
  PIDS+=("$!")
  LABELS+=("$label")
}

launch_edge_runs() {
  local depth
  local label
  for depth in 30 70; do
    label="edge_depth${depth}_2m_seed24061873_local${RUN_SUFFIX}"
    launch "$label" \
      N_PRIMARIES="$EDGE_PRIMARIES" RANDOM_SEED=24061873 \
      TARGET_DEPTH_MM="$depth" TARGET_THICKNESS_MM=1 \
      TARGET_LATERAL_MM=300 \
      bash run_local_export.sh edge "$label"
  done
}

launch_material_runs() {
  local label
  label="water_thin30_2m_seed24061873_local${RUN_SUFFIX}"
  launch "$label" \
    N_PRIMARIES="$MATERIAL_PRIMARIES" RANDOM_SEED=24061873 \
    TARGET_DEPTH_MM=30 TARGET_THICKNESS_MM=1 TARGET_LATERAL_MM=300 \
    TARGET_MATERIAL=Water bash run_local_export.sh thin_layer "$label"
  label="iron_thin30_2m_seed24061873_local${RUN_SUFFIX}"
  launch "$label" \
    N_PRIMARIES="$MATERIAL_PRIMARIES" RANDOM_SEED=24061873 \
    TARGET_DEPTH_MM=30 TARGET_THICKNESS_MM=1 TARGET_LATERAL_MM=300 \
    TARGET_MATERIAL=Iron bash run_local_export.sh thin_layer "$label"
}

case "$BATCH" in
  rods)
    label="rod_width1_depth30_5m_seed24061873_local${RUN_SUFFIX}"
    launch "$label" \
      N_PRIMARIES="$ROD_PRIMARIES" RANDOM_SEED=24061873 \
      TARGET_DEPTH_MM=30 TARGET_RADIUS_MM=0.5 TARGET_LATERAL_MM=300 \
      bash run_local_export.sh rod "$label"
    label="rod_width2_depth30_5m_seed24061873_local${RUN_SUFFIX}"
    launch "$label" \
      N_PRIMARIES="$ROD_PRIMARIES" RANDOM_SEED=24061873 \
      TARGET_DEPTH_MM=30 TARGET_RADIUS_MM=1 TARGET_LATERAL_MM=300 \
      bash run_local_export.sh rod "$label"
    label="rod_width5_depth30_5m_seed24061873_local${RUN_SUFFIX}"
    launch "$label" \
      N_PRIMARIES="$ROD_PRIMARIES" RANDOM_SEED=24061873 \
      TARGET_DEPTH_MM=30 TARGET_RADIUS_MM=2.5 TARGET_LATERAL_MM=300 \
      bash run_local_export.sh rod "$label"
    ;;
  double_rods_a)
    for separation in 5 10 15; do
      label="double_rod_width2_sep${separation}_depth30_5m_seed24061873_local${RUN_SUFFIX}"
      launch "$label" \
        N_PRIMARIES="$DOUBLE_ROD_PRIMARIES" RANDOM_SEED=24061873 \
        TARGET_DEPTH_MM=30 TARGET_RADIUS_MM=1 TARGET_LATERAL_MM=300 \
        TARGET_SEPARATION_MM="$separation" \
        bash run_local_export.sh double_rod "$label"
    done
    ;;
  double_rods_b)
    for separation in 20 30; do
      label="double_rod_width2_sep${separation}_depth30_5m_seed24061873_local${RUN_SUFFIX}"
      launch "$label" \
        N_PRIMARIES="$DOUBLE_ROD_PRIMARIES" RANDOM_SEED=24061873 \
        TARGET_DEPTH_MM=30 TARGET_RADIUS_MM=1 TARGET_LATERAL_MM=300 \
        TARGET_SEPARATION_MM="$separation" \
        bash run_local_export.sh double_rod "$label"
    done
    ;;
  edge)
    launch_edge_runs
    ;;
  materials)
    launch_material_runs
    ;;
  edge_material)
    launch_edge_runs
    launch_material_runs
    ;;
  void_pilot)
    for diameter in 10 20; do
      radius=$((diameter / 2))
      for depth in 30 60; do
        label="concrete_void_d${diameter}_depth${depth}_2m_seed24061875_local${RUN_SUFFIX}"
        launch "$label" \
          N_PRIMARIES="$VOID_PRIMARIES" RANDOM_SEED=24061875 \
          VOID_RADIUS_MM="$radius" VOID_LENGTH_MM=300 \
          VOID_X_MM=0 VOID_DEPTH_MM="$depth" \
          bash run_local_export.sh concrete_void "$label"
      done
    done
    ;;
  *)
    echo "Unknown campaign batch: $BATCH" >&2
    exit 2
    ;;
esac

status=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "Completed ${LABELS[$index]}"
  else
    exit_code=$?
    echo "Failed ${LABELS[$index]} (exit ${exit_code})" >&2
    status=1
  fi
done
exit "$status"
