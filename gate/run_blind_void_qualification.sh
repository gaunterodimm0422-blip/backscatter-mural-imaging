#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: bash run_blind_void_qualification.sh [campaign_tag]" >&2
  exit 2
fi

CAMPAIGN_TAG="${1:-blindq1}"
if [[ ! "$CAMPAIGN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "campaign_tag contains unsafe characters." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIVATE_ROOT="${BLIND_PRIVATE_ROOT:-/home/vgate/backscatter_blind_private}"
PUBLIC_ROOT="${BLIND_PUBLIC_ROOT:-/home/vgate/backscatter_blind_public}"
PRIVATE_CAMPAIGN="${PRIVATE_ROOT}/${CAMPAIGN_TAG}"
PUBLIC_CAMPAIGN="${PUBLIC_ROOT}/${CAMPAIGN_TAG}"
LOCAL_RUN_ROOT="${PRIVATE_CAMPAIGN}/simulation_runs"
CONTROLLER_ROOT="${PRIVATE_CAMPAIGN}/controller_logs"
UNUSED_EXPORT_ROOT="${PRIVATE_CAMPAIGN}/unused_export"
N_PRIMARIES="${BLIND_PRIMARIES:-1000000}"
MAX_PARALLEL="${BLIND_MAX_PARALLEL:-3}"

if [[ ! "$N_PRIMARIES" =~ ^[1-9][0-9]*$ ]]; then
  echo "BLIND_PRIMARIES must be a positive integer." >&2
  exit 2
fi
if [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
  echo "BLIND_MAX_PARALLEL must be a positive integer." >&2
  exit 2
fi
if (( MAX_PARALLEL > 3 )); then
  echo "BLIND_MAX_PARALLEL must not exceed 3 for this VM workflow." >&2
  exit 2
fi
if [[ -e "$PRIVATE_CAMPAIGN" || -e "$PUBLIC_CAMPAIGN" ]]; then
  echo "Refusing to overwrite an existing blind campaign: $CAMPAIGN_TAG" >&2
  echo "Choose a new campaign tag." >&2
  exit 2
fi

mkdir -p \
  "$PRIVATE_CAMPAIGN" \
  "$PUBLIC_CAMPAIGN" \
  "$LOCAL_RUN_ROOT" \
  "$CONTROLLER_ROOT" \
  "$UNUSED_EXPORT_ROOT"

# Create the answer key only inside the VM-private directory.  The public
# commitment binds the future reveal without exposing a brute-forceable
# unsalted hash of the small assignment space.
python3 - \
  "$PRIVATE_CAMPAIGN" \
  "$PUBLIC_CAMPAIGN" \
  "$CAMPAIGN_TAG" \
  "$N_PRIMARIES" <<'PY'
import datetime as dt
import hashlib
import json
import random
import secrets
import sys
from pathlib import Path

private_dir = Path(sys.argv[1])
public_dir = Path(sys.argv[2])
campaign_tag = sys.argv[3]
n_primaries = int(sys.argv[4])

assignment_seed = secrets.randbits(63)
rng = random.Random(assignment_seed)
assignments = [
    {"class": "void", "x_mm": x, "depth_mm": 30.0}
    for x in (-20.0, -10.0, 0.0, 0.0, 10.0, 20.0)
] + [
    {"class": "no_void", "x_mm": None, "depth_mm": None}
    for _ in range(6)
]
rng.shuffle(assignments)

cases = []
for index, assignment in enumerate(assignments, start=1):
    cases.append(
        {
            "case_id": f"{campaign_tag}_case{index:03d}",
            "class": assignment["class"],
            "void_diameter_mm": (
                20.0 if assignment["class"] == "void" else None
            ),
            "x_mm": assignment["x_mm"],
            "depth_mm": assignment["depth_mm"],
            "simulation_seed": 24061900 + index,
            "primary_events": n_primaries,
        }
    )

key = {
    "schema_version": "1.0",
    "campaign_tag": campaign_tag,
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "assignment_seed": assignment_seed,
    "protocol": {
        "positive_cases": 6,
        "negative_cases": 6,
        "positive_x_positions_mm": [-20.0, -10.0, 0.0, 0.0, 10.0, 20.0],
        "positive_depth_mm": 30.0,
        "void_diameter_mm": 20.0,
        "primary_events_per_case": n_primaries,
    },
    "cases": cases,
}
key_path = private_dir / "truth_key.json"
key_path.write_text(
    json.dumps(key, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
with (private_dir / "launch_plan.private.tsv").open(
    "w",
    encoding="utf-8",
    newline="",
) as handle:
    handle.write("case_id\tclass\tx_mm\tdepth_mm\tsimulation_seed\n")
    for case in cases:
        x_value = "NA" if case["x_mm"] is None else str(case["x_mm"])
        depth_value = (
            "NA" if case["depth_mm"] is None else str(case["depth_mm"])
        )
        handle.write(
            f"{case['case_id']}\t{case['class']}\t{x_value}\t"
            f"{depth_value}\t{case['simulation_seed']}\n"
        )

salt = secrets.token_bytes(32)
(private_dir / "truth_salt.txt").write_text(
    salt.hex() + "\n",
    encoding="ascii",
)
commitment = hashlib.sha256(salt + key_path.read_bytes()).hexdigest()
(public_dir / "truth_commitment.sha256").write_text(
    commitment + "\n",
    encoding="ascii",
)
(private_dir / "truth_commitment.sha256").write_text(
    commitment + "\n",
    encoding="ascii",
)
public_manifest = {
    "schema_version": "1.0",
    "campaign_tag": campaign_tag,
    "case_ids": [case["case_id"] for case in cases],
    "case_count": len(cases),
    "primary_events_per_case": n_primaries,
    "public_target_class": (
        "20 mm-diameter cylindrical void versus homogeneous concrete"
    ),
    "truth_commitment_sha256": commitment,
    "contains_case_labels_or_positions": False,
}
(public_dir / "campaign_public.json").write_text(
    json.dumps(public_manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

run_private_case() {
  local case_id="$1"
  local private_class="$2"
  local private_x_mm="$3"
  local private_depth_mm="$4"
  local simulation_seed="$5"
  local private_result="${LOCAL_RUN_ROOT}/${case_id}/gate/output/${case_id}"
  local public_staging="${PUBLIC_CAMPAIGN}/.${case_id}.partial"
  local public_final="${PUBLIC_CAMPAIGN}/${case_id}"

  if [[ "$private_class" == "void" ]]; then
    env \
      LOCAL_ROOT="$LOCAL_RUN_ROOT" \
      EXPORT_ROOT="$UNUSED_EXPORT_ROOT" \
      SKIP_EXPORT=1 \
      N_PRIMARIES="$N_PRIMARIES" \
      RANDOM_SEED="$simulation_seed" \
      VOID_RADIUS_MM=10 \
      VOID_LENGTH_MM=300 \
      VOID_X_MM="$private_x_mm" \
      VOID_DEPTH_MM="$private_depth_mm" \
      bash "${SCRIPT_DIR}/run_local_export.sh" concrete_void "$case_id"
  elif [[ "$private_class" == "no_void" ]]; then
    env \
      LOCAL_ROOT="$LOCAL_RUN_ROOT" \
      EXPORT_ROOT="$UNUSED_EXPORT_ROOT" \
      SKIP_EXPORT=1 \
      N_PRIMARIES="$N_PRIMARIES" \
      RANDOM_SEED="$simulation_seed" \
      bash "${SCRIPT_DIR}/run_local_export.sh" baseline "$case_id"
  else
    echo "Invalid private class for $case_id" >&2
    return 2
  fi

  for required in \
    hits.root \
    hits_truth_concrete.root \
    gate.log \
    run_parameters.txt \
    macro_sha256.txt \
    gate_version.txt
  do
    if [[ ! -s "${private_result}/${required}" ]]; then
      echo "Private case is incomplete: ${private_result}/${required}" >&2
      return 1
    fi
  done
  if ! grep -Eq \
    "Number of events processed[[:space:]]*:[[:space:]]*${N_PRIMARIES}" \
    "${private_result}/gate.log"
  then
    echo "Completion marker is absent for $case_id" >&2
    return 1
  fi
  if grep -Eq \
    "ERROR|FATAL|Aborted|core dumped|G4Exception" \
    "${private_result}/gate.log"
  then
    echo "Fatal marker found for $case_id" >&2
    return 1
  fi
  if [[ -e "$public_staging" || -e "$public_final" ]]; then
    echo "Refusing to overwrite a public case: $case_id" >&2
    return 2
  fi

  mkdir "$public_staging"
  cp "${private_result}/hits.root" "${public_staging}/hits.root"
  cp "${private_result}/gate_version.txt" "${public_staging}/gate_version.txt"
  cp "${private_result}/macro_sha256.txt" "${public_staging}/macro_sha256.txt"
  local private_hash
  local public_hash
  private_hash="$(sha256sum "${private_result}/hits.root" | awk '{print $1}')"
  public_hash="$(sha256sum "${public_staging}/hits.root" | awk '{print $1}')"
  if [[ "$private_hash" != "$public_hash" ]]; then
    echo "Private/public hits.root hash mismatch for $case_id" >&2
    return 1
  fi

  python3 - \
    "$public_staging" \
    "$case_id" \
    "$N_PRIMARIES" \
    "$public_hash" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
manifest = {
    "schema_version": "1.0",
    "case_id": sys.argv[2],
    "expected_primaries": int(sys.argv[3]),
    "hits_sha256": sys.argv[4].upper(),
    "gate_log_completion_verified_privately": True,
    "fatal_log_markers_absent_privately": True,
    "contains_truth_or_case_label": False,
    "public_bundle_created_utc": dt.datetime.now(
        dt.timezone.utc
    ).isoformat(),
}
(output_dir / "public_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
  mv "$public_staging" "$public_final"

  # The validated public hits.root is now the transfer source.  Remove only
  # the exact duplicate from the private run to limit VM disk use; keep the
  # private truth, log, parameters and macro hashes for post-reveal auditing.
  rm -f -- "${private_result}/hits.root"
}

declare -a PIDS=()
declare -a LABELS=()

wait_group() {
  local status=0
  local index
  for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
      echo "Completed public case ${LABELS[$index]}"
    else
      echo "Failed private case ${LABELS[$index]}" >&2
      status=1
    fi
  done
  PIDS=()
  LABELS=()
  return "$status"
}

while IFS=$'\t' read -r \
  case_id private_class private_x_mm private_depth_mm simulation_seed
do
  if [[ "$case_id" == "case_id" ]]; then
    continue
  fi
  echo "Launching blinded case ${case_id}"
  (
    run_private_case \
      "$case_id" \
      "$private_class" \
      "$private_x_mm" \
      "$private_depth_mm" \
      "$simulation_seed"
  ) >"${CONTROLLER_ROOT}/${case_id}.stdout.log" \
    2>"${CONTROLLER_ROOT}/${case_id}.stderr.log" &
  PIDS+=("$!")
  LABELS+=("$case_id")
  if (( ${#PIDS[@]} >= MAX_PARALLEL )); then
    wait_group
  fi
done < "${PRIVATE_CAMPAIGN}/launch_plan.private.tsv"

if (( ${#PIDS[@]} > 0 )); then
  wait_group
fi

public_count="$(
  find "$PUBLIC_CAMPAIGN" \
    -mindepth 1 \
    -maxdepth 1 \
    -type d \
    -name "${CAMPAIGN_TAG}_case*" |
    wc -l
)"
if [[ "$public_count" -ne 12 ]]; then
  echo "Expected 12 public cases, found ${public_count}." >&2
  exit 1
fi

echo
echo "Blind qualification simulation completed: 12/12 public cases."
echo "Public source: ${PUBLIC_CAMPAIGN}"
echo "Private answer key remains sealed in: ${PRIVATE_CAMPAIGN}"
echo "Do not run the reveal script until all score files are frozen and hashed."
