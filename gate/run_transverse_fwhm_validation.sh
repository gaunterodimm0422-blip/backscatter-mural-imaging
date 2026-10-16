#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-/home/vgate/backscatter_runs}"
CONTROLLER_ROOT="${CONTROLLER_ROOT:-/home/vgate/backscatter_controller}"
N_PRIMARIES="${N_PRIMARIES:-5000000}"
RANDOM_SEED="${RANDOM_SEED:-24080573}"
RUN_TAG="${RUN_TAG:-fwhm5_validation}"

if [[ ! "$N_PRIMARIES" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_PRIMARIES must be a positive integer." >&2
  exit 2
fi
if [[ ! "$RANDOM_SEED" =~ ^[1-9][0-9]*$ ]]; then
  echo "RANDOM_SEED must be a positive integer." >&2
  exit 2
fi
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "RUN_TAG must contain only letters, digits, dot, underscore, or hyphen." >&2
  exit 2
fi

if [[ "$N_PRIMARIES" == "5000000" ]]; then
  event_label="5m"
else
  event_label="${N_PRIMARIES}p"
fi
RUN_ID="rod_width2_depth30_${event_label}_seed${RANDOM_SEED}_local_${RUN_TAG}"
WORK_GATE="${LOCAL_ROOT}/${RUN_ID}/gate"
STDOUT_LOG="${CONTROLLER_ROOT}/${RUN_ID}.stdout.log"
STDERR_LOG="${CONTROLLER_ROOT}/${RUN_ID}.stderr.log"
PID_FILE="${CONTROLLER_ROOT}/${RUN_ID}.pid"

mkdir -p "$CONTROLLER_ROOT"
for path in "$WORK_GATE" "$STDOUT_LOG" "$STDERR_LOG" "$PID_FILE"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite an existing run or controller file: $path" >&2
    echo "Choose a new RUN_TAG; never append to an interrupted ROOT run." >&2
    exit 2
  fi
done

echo "Independent transverse-FWHM validation"
echo "  run ID:       $RUN_ID"
echo "  primaries:    $N_PRIMARIES"
echo "  GATE seed:    $RANDOM_SEED"
echo "  target:       2 mm concrete rod at 30 mm depth"
echo "  VM-local dir: $WORK_GATE"
echo
echo "Open a second terminal and run:"
echo "  cd $SCRIPT_DIR"
echo "  bash watch_transverse_fwhm_validation.sh $RUN_ID 30"
echo

cd "$SCRIPT_DIR"
(
  env \
    LOCAL_ROOT="$LOCAL_ROOT" \
    SKIP_EXPORT=1 \
    N_PRIMARIES="$N_PRIMARIES" \
    RANDOM_SEED="$RANDOM_SEED" \
    TARGET_DEPTH_MM=30 \
    TARGET_RADIUS_MM=1 \
    TARGET_LATERAL_MM=300 \
    bash run_local_export.sh rod "$RUN_ID"
) >"$STDOUT_LOG" 2>"$STDERR_LOG" &
worker_pid="$!"
printf '%s\n' "$worker_pid" >"$PID_FILE"

forward_signal() {
  local signal="$1"
  if kill -0 "$worker_pid" 2>/dev/null; then
    kill "-$signal" "$worker_pid" 2>/dev/null || true
  fi
}
trap 'forward_signal INT' INT
trap 'forward_signal TERM' TERM

if wait "$worker_pid"; then
  echo "Completed $RUN_ID"
  echo "Result remains VM-local until hash-validated transfer."
  exit 0
fi

exit_code="$?"
echo "Failed $RUN_ID (exit $exit_code)" >&2
if [[ -s "$STDERR_LOG" ]]; then
  echo "Last controller errors:" >&2
  tail -n 10 "$STDERR_LOG" >&2
fi
exit "$exit_code"
