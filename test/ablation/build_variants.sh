#!/bin/bash
# Build all 4 variants of libvgpu.so for ablation study
# Output: /tmp/libvgpu-{stock-delta,aimd,stock-delta-x3,aimd-v5}.so
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIBVGPU_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC="${LIBVGPU_ROOT}/src/multiprocess/multiprocess_utilization_watcher.c"
BACKUP="/tmp/watcher_backup.c"
cp "$SRC" "$BACKUP"

build_variant() {
    local name="$1"
    echo "=== Building: $name ==="
    docker run --rm \
      -e DEBIAN_FRONTEND=noninteractive -e TZ=UTC \
      -v "${LIBVGPU_ROOT}:/libvgpu:ro" \
      -v /tmp:/output \
      nvidia/cuda:12.2.0-devel-ubuntu20.04 \
      bash -c 'apt-get -y update > /dev/null 2>&1 && apt-get -y install cmake build-essential > /dev/null 2>&1 && mkdir -p /tmp/build && cd /tmp/build && cmake /libvgpu -DDLSYM_HOOK_ENABLE=1 -DMULTIPROCESS_LIMIT_ENABLE=1 -DHOOK_MEMINFO_ENABLE=1 -DHOOK_NVML_ENABLE=1 -DCMAKE_BUILD_TYPE=Debug > /dev/null 2>&1 && make -j$(nproc) 2>&1 | tail -3 && cp /tmp/build/libvgpu.so /output/libvgpu-'"$name"'.so && echo "BUILD OK: '"$name"'"'
}

patch_remove_x3() {
    # Remove ×3 from delta() increment
    sed -i 's|(long)utilization_diff / 2560 \* 3;|(long)utilization_diff / 2560;|' "$SRC"
}

patch_aimd_no_x3() {
    # Replace delta/change_token with AIMD (no ×3)
    python3 - "$SRC" << 'PYEOF'
import sys
src = open(sys.argv[1]).read()

old_block = """          /* Stock delta()/change_token() with ×3 increment scaling.
             Ablation test: isolate effect of ×3 correction from AIMD. */
          share = delta(upper_limit, userutil[0], share);
          change_token(share);"""

new_block = """          /* AIMD without ×3 correction (ablation). */
          long base = (long)g_sm_num * (long)g_max_thread_per_sm;
          int eff_limit = upper_limit * 7 / 8;
          long ai_step = base * (long)eff_limit / 400;
          if (userutil[0] <= eff_limit) {
            int gap = upper_limit - userutil[0];
            long step = ai_step * (long)(gap > 5 ? gap : 5) / 5;
            share = share + step;
          } else {
            share = share / 3;
          }
          if (share < ai_step) share = ai_step;
          long max_share = base * (long)eff_limit / 100;
          if (share > max_share) share = max_share;
          g_cur_cuda_cores = share;"""

if old_block not in src:
    print("ERROR: Could not find stock delta/change_token block in source", file=sys.stderr)
    sys.exit(1)
src = src.replace(old_block, new_block)
open(sys.argv[1], 'w').write(src)
PYEOF
}

patch_aimd_v5() {
    # Replace delta/change_token with AIMD + ×3
    python3 - "$SRC" << 'PYEOF'
import sys
src = open(sys.argv[1]).read()

old_block = """          /* Stock delta()/change_token() with ×3 increment scaling.
             Ablation test: isolate effect of ×3 correction from AIMD. */
          share = delta(upper_limit, userutil[0], share);
          change_token(share);"""

new_block = """          /* AIMD v5 with ×3 correction. */
          long base = (long)g_sm_num * (long)g_max_thread_per_sm * 3;
          int eff_limit = upper_limit * 7 / 8;
          long ai_step = base * (long)eff_limit / 400;
          if (userutil[0] <= eff_limit) {
            int gap = upper_limit - userutil[0];
            long step = ai_step * (long)(gap > 5 ? gap : 5) / 5;
            share = share + step;
          } else {
            share = share / 3;
          }
          if (share < ai_step) share = ai_step;
          long max_share = base * (long)eff_limit / 100;
          if (share > max_share) share = max_share;
          g_cur_cuda_cores = share;"""

if old_block not in src:
    print("ERROR: Could not find stock delta/change_token block in source", file=sys.stderr)
    sys.exit(1)
src = src.replace(old_block, new_block)
open(sys.argv[1], 'w').write(src)
PYEOF
}

# ============================================================
# Variant 1: stock-delta (original delta/change_token, no ×3)
# ============================================================
cp "$BACKUP" "$SRC"
patch_remove_x3
build_variant "stock-delta"

# ============================================================
# Variant 2: AIMD without ×3
# ============================================================
cp "$BACKUP" "$SRC"
patch_remove_x3
patch_aimd_no_x3
build_variant "aimd"

# ============================================================
# Variant 3: stock-delta + ×3 (current backup state)
# ============================================================
cp "$BACKUP" "$SRC"
build_variant "stock-delta-x3"

# ============================================================
# Variant 4: AIMD + ×3 (v5)
# ============================================================
cp "$BACKUP" "$SRC"
patch_remove_x3
patch_aimd_v5
build_variant "aimd-v5"

# Restore source to v5 state (best variant)
echo ""
echo "=== All builds complete ==="
ls -la /tmp/libvgpu-stock-delta.so /tmp/libvgpu-aimd.so /tmp/libvgpu-stock-delta-x3.so /tmp/libvgpu-aimd-v5.so
echo ""
echo "To deploy a variant:"
echo "  sudo cp /tmp/libvgpu-<variant>.so /usr/local/vgpu/libvgpu.so"
