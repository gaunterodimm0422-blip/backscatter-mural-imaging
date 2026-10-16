#!/usr/bin/env bash
set -u

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash watch_transverse_fwhm_validation.sh <run_id> [interval_seconds]" >&2
  exit 2
fi

RUN_ID="$1"
INTERVAL_SECONDS="${2:-30}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/vgate/backscatter_runs}"
CONTROLLER_ROOT="${CONTROLLER_ROOT:-/home/vgate/backscatter_controller}"

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "run_id contains unsafe characters." >&2
  exit 2
fi
if [[ ! "$INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "interval_seconds must be a positive integer." >&2
  exit 2
fi
if [[ "$RUN_ID" =~ _([0-9]+)p_seed ]]; then
  EXPECTED_PRIMARIES="${BASH_REMATCH[1]}"
else
  EXPECTED_PRIMARIES="5000000"
fi

OUTPUT_DIR="${LOCAL_ROOT}/${RUN_ID}/gate/output/${RUN_ID}"
GATE_LOG="${OUTPUT_DIR}/gate.log"
STDERR_LOG="${CONTROLLER_ROOT}/${RUN_ID}.stderr.log"
PID_FILE="${CONTROLLER_ROOT}/${RUN_ID}.pid"

while true; do
  printf '\033[2J\033[H'
  date --iso-8601=seconds
  echo "Run: $RUN_ID"
  echo "Expected primaries: $EXPECTED_PRIMARIES"
  echo

  size="0"
  if [[ -d "$OUTPUT_DIR" ]]; then
    size="$(du -sh "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')"
  fi
  echo "Current output size: $size"
  df -h /home 2>/dev/null | awk 'NR == 1 || NR == 2 {print}'
  echo

  worker_running=0
  if [[ -s "$PID_FILE" ]]; then
    worker_pid="$(tr -d '[:space:]' <"$PID_FILE")"
    if [[ "$worker_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$worker_pid" 2>/dev/null; then
      worker_running=1
      ps -p "$worker_pid" -o pid=,etime=,%cpu=,%mem=,cmd= 2>/dev/null || true
    fi
  fi

  fatal=0
  complete=0
  if [[ -s "$GATE_LOG" ]]; then
    if grep -Eq 'ERROR|FATAL|Aborted|core dumped|G4Exception' "$GATE_LOG"; then
      fatal=1
    fi
    if grep -Eq "Number of events processed[[:space:]]*:[[:space:]]*${EXPECTED_PRIMARIES}([[:space:]]|$)" "$GATE_LOG"; then
      complete=1
    fi
  fi

  if [[ "$fatal" -eq 1 ]]; then
    echo
    echo "Status: FAILED (fatal marker in gate.log)"
    tail -n 12 "$GATE_LOG"
    exit 1
  fi
  if [[ "$complete" -eq 1 ]]; then
    echo
    echo "Status: GATE COMPLETED"
    echo "The exact ${EXPECTED_PRIMARIES}-event completion marker is present."
    echo "Next: hash-validate, transfer, and run the frozen 5 mm analysis."
    exit 0
  fi
  if [[ "$worker_running" -eq 0 ]]; then
    echo
    echo "Status: NOT RUNNING OR INTERRUPTED"
    if [[ -s "$STDERR_LOG" ]]; then
      echo "Last controller errors:"
      tail -n 10 "$STDERR_LOG"
    fi
    exit 1
  fi

  echo
  echo "Status: RUNNING"
  echo "ROOT size is only a progress proxy; acceptance uses the exact event marker."
  if [[ -s "$STDERR_LOG" ]]; then
    echo "Recent controller stderr:"
    tail -n 3 "$STDERR_LOG"
  fi
  sleep "$INTERVAL_SECONDS"
done
