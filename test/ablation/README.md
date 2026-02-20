# AIMD×3 Patch for HAMi gpucores Controller

Minimal patch to the HAMi-core `utilization_watcher()` that reduces SM utilization
control error from **MAE 20.7%** (stock) to **MAE 2.3%**.

## Branches

| Branch | Description |
|--------|-------------|
| `main` | Upstream HAMi-core v2.8.0 (unmodified) |
| `ablation/orig-aimd-v5` | **Recommended**: AIMD×3 patch (15 lines changed in 1 file) |

## What Changed

Only `src/multiprocess/multiprocess_utilization_watcher.c` is modified:

```diff
-          share = delta(upper_limit, userutil[0], share);
-          change_token(share);
+          /* AIMD v5 with ×3 correction (no NVML fixes). */
+          long base = (long)g_sm_num * (long)g_max_thread_per_sm * 3;
+          int eff_limit = upper_limit * 7 / 8;
+          long ai_step = base * (long)eff_limit / 400;
+          if (userutil[0] <= eff_limit) {
+            int gap = upper_limit - userutil[0];
+            long step = ai_step * (long)(gap > 5 ? gap : 5) / 5;
+            share = share + step;
+          } else {
+            share = share / 3;
+          }
+          if (share < ai_step) share = ai_step;
+          long max_share = base * (long)eff_limit / 100;
+          if (share > max_share) share = max_share;
+          g_cur_cuda_cores = share;
```

**Why it works** — two essential components:
1. **AIMD**: Additive increase, multiplicative decrease (÷3). Direct assignment
   `g_cur_cuda_cores = share` instead of additive `change_token()`.
2. **×3 scaling**: `base = SM × threads × 3` compensates for token-to-throughput
   scale mismatch. Without it, AIMD undershoots (MAE 33%).

## Build → Install → Test

### Prerequisites

- Docker (with `nvidia/cuda:12.2.0-devel-ubuntu20.04` image)
- k3s cluster with [HAMi](https://github.com/Project-HAMi/HAMi) device plugin
- `nvidia-smi` accessible on the host
- `kubectl` configured for the cluster

### 1. Build

`libvgpu.so` must be built inside Docker (Ubuntu 20.04, glibc 2.31) because it
is injected via `LD_PRELOAD` into GPU containers that use older base images.
Building on the host (Ubuntu 24.04, glibc 2.39) produces incompatible binaries.

```bash
# From the libvgpu repo root:
git checkout ablation/orig-aimd-v5
bash test/ablation/build.sh              # → /tmp/libvgpu.so
bash test/ablation/build.sh /tmp/my.so   # custom output path
```

### 2. Install

```bash
# Back up the current binary (first time only):
sudo cp /usr/local/vgpu/libvgpu.so /usr/local/vgpu/libvgpu.so.backup

# Deploy the new binary:
sudo cp /tmp/libvgpu.so /usr/local/vgpu/libvgpu.so

# Verify:
md5sum /usr/local/vgpu/libvgpu.so
```

No restart of the HAMi device plugin or k3s is required — `libvgpu.so` is loaded
via `LD_PRELOAD` at container startup, so the new binary takes effect on the next
Pod creation.

### 3. Test (single level)

```bash
# Run gpu_burn for 30s at gpucores=40:
bash test/ablation/k3s_collect.sh test 40

# Output:
#   /tmp/gpu-bench-ts/k3s/test_sm40_gpuburn.log
#   /tmp/gpu-bench-ts/k3s/test_sm40_smi.csv
```

### 4. Test (full sweep)

```bash
# Sweep across gpucores = 0, 20, 40, 60, 80:
bash test/ablation/k3s_sweep.sh test "0 20 40 60 80"
```

### 5. Plot results

```bash
# Final comparison (requires both Original and AIMD×3 data):
python3 test/ablation/plot_final_comparison.py
# → /tmp/gpu-bench-ts/k3s_final_comparison.png
```

## Test Scripts

| File | Purpose |
|------|---------|
| `build.sh` | Build `libvgpu.so` from current branch via Docker |
| `k3s_collect.sh` | Run one gpu_burn benchmark (30s) and collect data |
| `k3s_sweep.sh` | Sweep across multiple gpucores levels |
| `plot_final_comparison.py` | 2-variant comparison plot (Original vs AIMD×3) |

### Ablation-only scripts (historical)

These scripts were used for the 6-variant ablation study. They reference branches
that have been deleted, but the scripts are preserved for reference.

| File | Purpose |
|------|---------|
| `build_variants.sh` | Build all 4 ablation variant binaries |
| `run_ablation.sh` | Run full ablation sweep with manual binary deployment |
| `plot_ablation.py` | 6-variant ablation comparison plots |

## Results (RTX 4080 SUPER, k3s, gpu_burn 30s, 2 runs each)

| Variant | sm=0 | sm=20 | sm=40 | sm=60 | sm=80 | MAE |
|---------|------|-------|-------|-------|-------|-----|
| Original (v2.8.0) | 12474±35 | 40.2% | 64.7% | 83.6% | 95.2% | **20.7%** |
| Orig + AIMD×3 | 12443±50 | 20.8% | 41.1% | 58.7% | 74.4% | **2.3%** |

## Ablation Study Summary

Full 6-variant ablation was conducted to isolate the contribution of each change:

| Variant | MAE | Key finding |
|---------|-----|-------------|
| Original (v2.8.0) | 20.7% | Baseline — massive overshoot |
| Stock + NVML fixes | 23.9% | NVML fixes alone WORSEN stock |
| AIMD (no ×3) | 33.1% | AIMD alone — massive undershoot |
| Stock + ×3 | 21.9% | ×3 alone — no improvement |
| **Orig + AIMD×3** | **2.3%** | **Minimal patch, sufficient** |
| AIMD v5 (×3+NVML) | 0.9% | Full fixes, marginal improvement |

## Data Format

### gpu_burn log

Each line contains progress percentage and cumulative proc'd count:
```
16.67% proc'd: 2081 (3568 Gflop/s)
```

### nvidia-smi CSV

```csv
timestamp,utilization_gpu
1708123456.789,45
```

## Notes

- `GPU_CORE_UTILIZATION_POLICY=FORCE` must be set in the container for
  `rate_limiter()` to activate (bypasses shared memory gate).
- The ×3, /400, /3 constants were tuned on RTX 4080 SUPER + gpu_burn.
  Different GPUs or workloads (e.g., vLLM) may require adjustment.
