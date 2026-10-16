#!/usr/bin/env bash
set -u

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash watch_rods.sh <run_tag> [interval_seconds]" >&2
  exit 2
fi

RUN_TAG="$1"
INTERVAL_SECONDS="${2:-30}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/vgate/backscatter_runs}"
CONTROLLER_ROOT="${CONTROLLER_ROOT:-/home/vgate/backscatter_controller}"

if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "run_tag must contain only letters, digits, dot, underscore, or hyphen." >&2
  exit 2
fi
if [[ ! "$INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "interval_seconds must be a positive integer." >&2
  exit 2
fi

RUN_IDS=(
  "rod_width1_depth30_5m_seed24061873_local_${RUN_TAG}"
  "rod_width2_depth30_5m_seed24061873_local_${RUN_TAG}"
  "rod_width5_depth30_5m_seed24061873_local_${RUN_TAG}"
)

while true; do
  printf '\033[2J\033[H'
  date --iso-8601=seconds
  gate_count="$(pgrep -c -x Gate 2>/dev/null || true)"
  gate_count="${gate_count:-0}"
  echo "Active GATE processes: ${gate_count}"
  echo

  complete_count=0
  for run_id in "${RUN_IDS[@]}"; do
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

    printf '%-58s  %-22s  %8s\n' "$run_id" "$status" "$size"
    if [[ -s "$stderr_log" ]]; then
      echo "  stderr:"
      tail -n 3 "$stderr_log" | sed 's/^/    /'
    fi
  done

  echo
  echo "Completed: ${complete_count}/3"
  echo "ROOT directory size is only a progress proxy; final acceptance requires"
  echo "the 5,000,000-event completion marker and later hash validation."

  if [[ "$complete_count" -eq 3 ]]; then
    echo "All rod runs completed. Tell Codex to validate and transfer them."
    exit 0
  fi
  if [[ "$gate_count" -eq 0 ]]; then
    echo "No GATE process is active. The batch is either still starting or was interrupted."
    echo "Check the first terminal and rerun this watcher after the launch messages."
    exit 1
  fi

  sleep "$INTERVAL_SECONDS"
done
