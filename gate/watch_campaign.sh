#!/usr/bin/env bash
set -u

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash watch_campaign.sh <batch> <run_tag> [interval_seconds]" >&2
  echo "Batches: void_pilot, double_rods_a, double_rods_b, edge, materials, edge_material" >&2
  exit 2
fi

BATCH="$1"
RUN_TAG="$2"
INTERVAL_SECONDS="${3:-30}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/vgate/backscatter_runs}"
CONTROLLER_ROOT="${CONTROLLER_ROOT:-/home/vgate/backscatter_controller}"
DOUBLE_ROD_PRIMARIES="${DOUBLE_ROD_PRIMARIES:-5000000}"
EDGE_PRIMARIES="${EDGE_PRIMARIES:-2000000}"
MATERIAL_PRIMARIES="${MATERIAL_PRIMARIES:-2000000}"
VOID_PRIMARIES="${VOID_PRIMARIES:-2000000}"

if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "run_tag must contain only letters, digits, dot, underscore, or hyphen." >&2
  exit 2
fi
if [[ ! "$INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "interval_seconds must be a positive integer." >&2
  exit 2
fi
for variable_name in \
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

RUN_SUFFIX="_${RUN_TAG}"
declare -a RUN_IDS=()
declare -a EXPECTED_COUNTS=()

add_run() {
  RUN_IDS+=("$1")
  EXPECTED_COUNTS+=("$2")
}

case "$BATCH" in
  void_pilot)
    for diameter in 10 20; do
      for depth in 30 60; do
        add_run \
          "concrete_void_d${diameter}_depth${depth}_2m_seed24061875_local${RUN_SUFFIX}" \
          "$VOID_PRIMARIES"
      done
    done
    ;;
  double_rods_a)
    for separation in 5 10 15; do
      add_run \
        "double_rod_width2_sep${separation}_depth30_5m_seed24061873_local${RUN_SUFFIX}" \
        "$DOUBLE_ROD_PRIMARIES"
    done
    ;;
  double_rods_b)
    for separation in 20 30; do
      add_run \
        "double_rod_width2_sep${separation}_depth30_5m_seed24061873_local${RUN_SUFFIX}" \
        "$DOUBLE_ROD_PRIMARIES"
    done
    ;;
  edge)
    for depth in 30 70; do
      add_run \
        "edge_depth${depth}_2m_seed24061873_local${RUN_SUFFIX}" \
        "$EDGE_PRIMARIES"
    done
    ;;
  materials)
    add_run \
      "water_thin30_2m_seed24061873_local${RUN_SUFFIX}" \
      "$MATERIAL_PRIMARIES"
    add_run \
      "iron_thin30_2m_seed24061873_local${RUN_SUFFIX}" \
      "$MATERIAL_PRIMARIES"
    ;;
  edge_material)
    for depth in 30 70; do
      add_run \
        "edge_depth${depth}_2m_seed24061873_local${RUN_SUFFIX}" \
        "$EDGE_PRIMARIES"
    done
    add_run \
      "water_thin30_2m_seed24061873_local${RUN_SUFFIX}" \
      "$MATERIAL_PRIMARIES"
    add_run \
      "iron_thin30_2m_seed24061873_local${RUN_SUFFIX}" \
      "$MATERIAL_PRIMARIES"
    ;;
  *)
    echo "Unknown campaign batch: $BATCH" >&2
    exit 2
    ;;
esac

total_count="${#RUN_IDS[@]}"
while true; do
  printf '\033[2J\033[H'
  date --iso-8601=seconds
  gate_count="$(pgrep -c -x Gate 2>/dev/null || true)"
  gate_count="${gate_count:-0}"
  echo "Campaign: ${BATCH}"
  echo "Active GATE processes: ${gate_count}"
  echo

  complete_count=0
  for index in "${!RUN_IDS[@]}"; do
    run_id="${RUN_IDS[$index]}"
    expected="${EXPECTED_COUNTS[$index]}"
    output_dir="${LOCAL_ROOT}/${run_id}/gate/output/${run_id}"
    stdout_log="${CONTROLLER_ROOT}/${run_id}.stdout.log"
    stderr_log="${CONTROLLER_ROOT}/${run_id}.stderr.log"
    status="waiting for local output"
    size="0"

    if [[ -d "$output_dir" ]]; then
      size="$(du -sh "$output_dir" 2>/dev/null | awk '{print $1}')"
      status="running or interrupted"
    fi
    if [[ -f "$stdout_log" ]] &&
       grep -Fq "Completed: output/${run_id}" "$stdout_log"; then
      status="GATE completed"
      complete_count=$((complete_count + 1))
    fi

    printf '%-72s  %-22s  %8s  expected=%s\n' \
      "$run_id" "$status" "$size" "$expected"
    if [[ -s "$stderr_log" ]]; then
      echo "  stderr:"
      tail -n 3 "$stderr_log" | sed 's/^/    /'
    fi
  done

  echo
  echo "Completed: ${complete_count}/${total_count}"
  echo "Directory size is only a progress proxy. Final acceptance requires each"
  echo "run's expected-event completion marker and later SHA-256 validation."

  if [[ "$complete_count" -eq "$total_count" ]]; then
    echo "All runs completed. Tell Codex to validate and transfer them."
    exit 0
  fi
  if [[ "$gate_count" -eq 0 ]]; then
    echo "No GATE process is active. The batch is either still starting or was interrupted."
    echo "Check the launch terminal and controller stderr logs."
    exit 1
  fi

  sleep "$INTERVAL_SECONDS"
done
