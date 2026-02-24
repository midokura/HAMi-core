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

### Why Docker?

`libvgpu.so` is injected via `LD_PRELOAD` into GPU containers that use older base
images (Ubuntu 20.04, glibc 2.31). Building on the host (Ubuntu 24.04, glibc 2.39)
produces incompatible binaries, so the build runs inside Docker.

### Quick test (single level)

```bash
git checkout ablation/orig-aimd-v5
bash test/ablation/build.sh                        # → /tmp/libvgpu.so
sudo cp /usr/local/vgpu/libvgpu.so /usr/local/vgpu/libvgpu.so.backup
sudo cp /tmp/libvgpu.so /usr/local/vgpu/libvgpu.so
bash test/ablation/k3s_collect.sh test 0            # baseline (no gpucores limit)
bash test/ablation/k3s_collect.sh test 40           # gpu_burn 30s at gpucores=40
python3 test/ablation/plot_single.py test 40        # → /tmp/gpu-bench-ts/k3s/test_sm40_plot.png
```

`plot_single.py` automatically uses `test_sm0` as the baseline. You can also
specify it manually: `python3 test/ablation/plot_single.py test 40 --baseline 12443`

No restart of the HAMi device plugin or k3s is needed — `libvgpu.so` is loaded
via `LD_PRELOAD` at container startup, so the new binary takes effect on the next
Pod creation.

### Full comparison (Original vs AIMD×3)

The plot script expects data with specific labels. Follow these steps to collect
data for both variants and generate the comparison plot.

```bash
# ── Step 1: Build and deploy Original (main branch) ──
git checkout main
bash test/ablation/build.sh /tmp/libvgpu-stock.so
sudo cp /tmp/libvgpu-stock.so /usr/local/vgpu/libvgpu.so

# ── Step 2: Collect Original data (2 runs for error bars) ──
for sm in 0 20 40 60 80; do bash test/ablation/k3s_collect.sh stock  $sm; done
for sm in 0 20 40 60 80; do bash test/ablation/k3s_collect.sh stock2 $sm; done

# ── Step 3: Build and deploy AIMD×3 (patched branch) ──
git checkout ablation/orig-aimd-v5
bash test/ablation/build.sh /tmp/libvgpu-patched.so
sudo cp /tmp/libvgpu-patched.so /usr/local/vgpu/libvgpu.so

# ── Step 4: Collect AIMD×3 data (2 runs for error bars) ──
for sm in 0 20 40 60 80; do bash test/ablation/k3s_collect.sh origv5  $sm; done
for sm in 0 20 40 60 80; do bash test/ablation/k3s_collect.sh origv5b $sm; done

# ── Step 5: Generate comparison plot ──
python3 test/ablation/plot_final_comparison.py
# → /tmp/gpu-bench-ts/k3s_final_comparison.png
```

Data is saved to `/tmp/gpu-bench-ts/k3s/`. Each run produces two files:
- `<label>_sm<N>_gpuburn.log` — gpu_burn throughput log
- `<label>_sm<N>_smi.csv` — nvidia-smi SM utilization at 100ms intervals

## Scripts

| File | Purpose |
|------|---------|
| `build.sh` | Build `libvgpu.so` from current branch via Docker |
| `k3s_collect.sh` | Run one gpu_burn benchmark (30s) and collect gpu_burn log + nvidia-smi CSV |
| `plot_single.py` | Plot a single run (`python3 plot_single.py <label> <sm>`) |
| `plot_final_comparison.py` | Generate comparison plot (expects labels: `stock`, `stock2`, `origv5`, `origv5b`) |

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

## Notes

- The benchmark scripts set `GPU_CORE_UTILIZATION_POLICY=FORCE` to bypass the
  shared memory gate. In normal HAMi deployments, the device plugin sets
  `utilization_switch` automatically when `gpucores` is configured, so `FORCE`
  is not needed for production workloads.
- The ×3, /400, /3 constants were tuned on RTX 4080 SUPER + gpu_burn.
  Different GPUs or workloads (e.g., vLLM) may require adjustment.
