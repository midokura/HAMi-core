#!/bin/bash
# Run full ablation study: sweep all 4 variants
# Requires manual binary deployment between variants.
#
# Usage: bash run_ablation.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIBPATH="/usr/local/vgpu/libvgpu.so"
LEVELS="0 20 40 60 80"

VARIANTS=("stock-delta:sd" "aimd:aimd" "stock-delta-x3:sdx3" "aimd-v5:v5")

for entry in "${VARIANTS[@]}"; do
    IFS=':' read -r binary_name label <<< "$entry"
    BINARY="/tmp/libvgpu-${binary_name}.so"

    echo ""
    echo "################################################################"
    echo "# Variant: ${binary_name} (label: ${label})"
    echo "################################################################"

    if [ ! -f "$BINARY" ]; then
        echo "ERROR: Binary not found: $BINARY"
        echo "Run build_variants.sh first."
        exit 1
    fi

    echo "Deploy: sudo cp ${BINARY} ${LIBPATH}"
    read -p "Press Enter after deploying the binary..."

    bash "${SCRIPT_DIR}/k3s_sweep.sh" "$label" "$LEVELS"
done

# Restore v5
echo ""
echo "################################################################"
echo "# Restore AIMD v5 as active binary"
echo "################################################################"
echo "Deploy: sudo cp /tmp/libvgpu-aimd-v5.so ${LIBPATH}"
read -p "Press Enter after restoring v5..."

echo ""
echo "=== Ablation complete ==="
echo "Generate comparison plot:"
echo "  python3 ${SCRIPT_DIR}/plot_ablation.py"
