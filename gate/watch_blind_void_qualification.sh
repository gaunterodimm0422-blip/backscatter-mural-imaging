#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: bash watch_blind_void_qualification.sh [campaign_tag]" >&2
  exit 2
fi

CAMPAIGN_TAG="${1:-blindq1}"
if [[ ! "$CAMPAIGN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "campaign_tag contains unsafe characters." >&2
  exit 2
fi

PUBLIC_ROOT="${BLIND_PUBLIC_ROOT:-/home/vgate/backscatter_blind_public}"
PUBLIC_CAMPAIGN="${PUBLIC_ROOT}/${CAMPAIGN_TAG}"
active_count="$(pgrep -x Gate 2>/dev/null | wc -l || true)"
completed_count=0

echo "$(date --iso-8601=seconds)"
echo "Active GATE processes: ${active_count}"
echo
for index in $(seq -w 1 12); do
  case_id="${CAMPAIGN_TAG}_case0${index}"
  case_dir="${PUBLIC_CAMPAIGN}/${case_id}"
  if [[ -s "${case_dir}/hits.root" && -s "${case_dir}/public_manifest.json" ]]; then
    size="$(du -h "${case_dir}/hits.root" | awk '{print $1}')"
    printf "%-34s  %-12s  %s\n" "$case_id" "public ready" "$size"
    completed_count=$((completed_count + 1))
  else
    printf "%-34s  %s\n" "$case_id" "running, queued, or interrupted"
  fi
done
echo
echo "Public cases completed: ${completed_count}/12"
echo "The monitor intentionally does not display private class or position."
df -h /home/vgate | awk 'NR == 1 || NR == 2'
