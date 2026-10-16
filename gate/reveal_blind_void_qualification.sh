#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: SCORE_BUNDLE_SHA256=<64 hex> bash reveal_blind_void_qualification.sh <campaign_tag> [export_root]" >&2
  exit 2
fi

CAMPAIGN_TAG="$1"
EXPORT_ROOT="${2:-/media/sf_share/beisanshe_v1/blind_reveal}"
PRIVATE_ROOT="${BLIND_PRIVATE_ROOT:-/home/vgate/backscatter_blind_private}"
PUBLIC_ROOT="${BLIND_PUBLIC_ROOT:-/home/vgate/backscatter_blind_public}"
PRIVATE_CAMPAIGN="${PRIVATE_ROOT}/${CAMPAIGN_TAG}"
PUBLIC_CAMPAIGN="${PUBLIC_ROOT}/${CAMPAIGN_TAG}"
FINAL_DIR="${EXPORT_ROOT}/${CAMPAIGN_TAG}"
STAGING_DIR="${EXPORT_ROOT}/.${CAMPAIGN_TAG}.partial"
SCORE_HASH="${SCORE_BUNDLE_SHA256:-}"
COMMITMENT_PATH="${PUBLIC_CAMPAIGN}/truth_commitment.sha256"
if [[ ! -s "$COMMITMENT_PATH" ]]; then
  COMMITMENT_PATH="${PRIVATE_CAMPAIGN}/truth_commitment.sha256"
fi

if [[ ! "$CAMPAIGN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "campaign_tag contains unsafe characters." >&2
  exit 2
fi
if [[ ! "$SCORE_HASH" =~ ^[A-Fa-f0-9]{64}$ ]]; then
  echo "A frozen 64-character SCORE_BUNDLE_SHA256 is required before reveal." >&2
  exit 2
fi
for required in \
  "${PRIVATE_CAMPAIGN}/truth_key.json" \
  "${PRIVATE_CAMPAIGN}/truth_salt.txt" \
  "$COMMITMENT_PATH"
do
  if [[ ! -s "$required" ]]; then
    echo "Missing blind commitment file: $required" >&2
    exit 1
  fi
done
if [[ -e "$FINAL_DIR" || -e "$STAGING_DIR" ]]; then
  echo "Refusing to overwrite an existing reveal export." >&2
  exit 2
fi

python3 - \
  "${PRIVATE_CAMPAIGN}/truth_key.json" \
  "${PRIVATE_CAMPAIGN}/truth_salt.txt" \
  "$COMMITMENT_PATH" <<'PY'
import hashlib
import sys
from pathlib import Path

key_path = Path(sys.argv[1])
salt = bytes.fromhex(Path(sys.argv[2]).read_text(encoding="ascii").strip())
expected = Path(sys.argv[3]).read_text(encoding="ascii").strip().lower()
actual = hashlib.sha256(salt + key_path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit("Truth-key commitment verification failed")
PY

mkdir -p "$EXPORT_ROOT"
mkdir "$STAGING_DIR"
cp "${PRIVATE_CAMPAIGN}/truth_key.json" "$STAGING_DIR/"
cp "${PRIVATE_CAMPAIGN}/truth_salt.txt" "$STAGING_DIR/"
cp "$COMMITMENT_PATH" "$STAGING_DIR/truth_commitment.sha256"
printf "%s\n" "${SCORE_HASH^^}" > "${STAGING_DIR}/score_bundle_sha256.txt"
mv "$STAGING_DIR" "$FINAL_DIR"

echo "Blind answer key revealed after score commitment."
echo "Reveal bundle: ${FINAL_DIR}"
echo "Do not edit the score bundle or reveal files before host verification."
