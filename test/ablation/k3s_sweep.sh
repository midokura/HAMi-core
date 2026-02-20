#!/bin/bash
# Run time-series collection for multiple gpucores levels
#
# Usage: bash k3s_sweep.sh <label> [levels]
#   label   - variant label (e.g., "v5", "sd", "aimd", "sdx3")
#   levels  - space-separated gpucores values (default: "0 20 40 60 80")
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL=${1:-"v5"}
LEVELS=${2:-"0 20 40 60 80"}

echo "=== k3s time-series sweep: ${LABEL} ==="
echo "Levels: ${LEVELS}"
echo ""

for SM in $LEVELS; do
  bash "${SCRIPT_DIR}/k3s_collect.sh" "$LABEL" "$SM"
  echo ""
  sleep 5
done

echo "=== All done ==="
ls -la /tmp/gpu-bench-ts/k3s/${LABEL}_*
