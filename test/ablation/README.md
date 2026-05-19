# AIMD×3 Patch for HAMi gpucores Controller

Minimal patch to the HAMi-core `utilization_watcher()` that reduces SM utilization
control error from **MAE 20.7%** (stock) to **MAE 2.3%**.

## Branches

| Branch | Description |
|--------|-------------|
| `main` | Upstream HAMi-core v2.8.0 (unmodified, no test scripts) |
| `ablation/stock` | `main` + test scripts (for benchmarking the stock controller) |
| `ablation/orig-aimd-v5` | **Recommended**: AIMD×3 patch (15 lines changed in 1 file) + test scripts |

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

### ⚠️ Critical: deploy target is `libvgpu.so.v2.8.0`, NOT `libvgpu.so`

On HAMi v2.8.0 the device plugin bind-mounts the **version-specific** file into
pods:

```
host  /usr/local/vgpu/libvgpu.so.v2.8.0   →   pod  /usr/local/vgpu/libvgpu.so   (ro bind mount)
```

So overwriting host `libvgpu.so` has no effect — pods keep loading the vanilla
`libvgpu.so.v2.8.0` baked into the HAMi image. **The patch must be deployed to
`libvgpu.so.v2.8.0`** (or to whatever versioned file name HAMi installs for the
current chart version).

Verify with:

```bash
# host side
md5sum /usr/local/vgpu/libvgpu.so.v2.8.0
# inside a running pod
kubectl exec <gpu-pod> -- md5sum /usr/local/vgpu/libvgpu.so
# both should match the patched build's md5
cat /proc/$(kubectl get pod <gpu-pod> -o jsonpath='{.status.containerStatuses[0].containerID}' | cut -d/ -f3)/mountinfo \
    2>/dev/null | grep libvgpu
# expect a line containing  libvgpu.so.v2.8.0  ...  /usr/local/vgpu/libvgpu.so
```

### Quick test (single level)

```bash
git checkout ablation/orig-aimd-v5
bash test/ablation/build.sh                                # → /tmp/libvgpu.so

# back up the vanilla v2.8.0 once (idempotent)
sudo cp -n /usr/local/vgpu/libvgpu.so.v2.8.0 /usr/local/vgpu/libvgpu.so.v2.8.0.stock

# deploy patch to the bind-mount source, NOT to libvgpu.so
sudo cp /tmp/libvgpu.so /usr/local/vgpu/libvgpu.so.v2.8.0

bash test/ablation/k3s_collect.sh test 0            # baseline (no gpucores limit)
bash test/ablation/k3s_collect.sh test 40           # gpu_burn 30s at gpucores=40
python3 test/ablation/plot_single.py test 40        # → /tmp/gpu-bench-ts/k3s/test_sm40_plot.png
```

`plot_single.py` automatically uses `test_sm0` as the baseline. You can also
specify it manually: `python3 test/ablation/plot_single.py test 40 --baseline 12443`

No restart of the HAMi device plugin or k3s is needed — `libvgpu.so` is loaded
via `LD_PRELOAD` at container startup, so the new binary takes effect on the next
Pod creation.

### Sanity check: confirm the AIMD code path is actually executing

Before trusting any results, run a probe pod with verbose logging and confirm
the patched watcher is in use (not the vanilla one):

```bash
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata: { name: gpu-probe }
spec:
  restartPolicy: Never
  containers:
    - name: bench
      image: oguzpastirmaci/gpu-burn
      command: ["/bin/sh","-c","export LIBCUDA_LOG_LEVEL=4 GPU_CORE_UTILIZATION_POLICY=FORCE; md5sum /usr/local/vgpu/libvgpu.so; ./gpu_burn 25"]
      resources:
        limits:
          nvidia.com/gpu: 1
          nvidia.com/gpumem: 3000
          nvidia.com/gpucores: 40
EOF
kubectl wait --for=condition=Ready pod/gpu-probe --timeout=60s
# wait for completion, then:
kubectl logs gpu-probe > /tmp/probe.log
grep -c   "delta:"      /tmp/probe.log    # expect 0 (AIMD removes the change_token() call)
grep -m1  "userutil1="  /tmp/probe.log    # expect line number ~222 (patched source);
                                          # vanilla / stock prints it at ~209
kubectl delete pod gpu-probe
```

If `delta:` count is non-zero or `userutil1=` references line ~209, the pod is
loading the vanilla binary — re-check the deploy target.

### Full comparison (Original vs AIMD×3)

The plot script expects data with specific labels. Follow these steps to collect
data for both variants and generate the comparison plot.

> All `sudo cp` targets below are **`libvgpu.so.v2.8.0`** — see the warning at
> the top of this file for why `libvgpu.so` is wrong.

```bash
# ── Step 0: One-time backup of the vanilla HAMi v2.8.0 binary ──
sudo cp -n /usr/local/vgpu/libvgpu.so.v2.8.0 /usr/local/vgpu/libvgpu.so.v2.8.0.stock

# ── Step 1: Restore vanilla (Original = HAMi v2.8.0 untouched) ──
sudo cp /usr/local/vgpu/libvgpu.so.v2.8.0.stock /usr/local/vgpu/libvgpu.so.v2.8.0

# ── Step 2: Collect Original data (2 runs for error bars) ──
for sm in 0 20 40 60 80; do bash test/ablation/k3s_collect.sh stock  $sm; done
for sm in 0 20 40 60 80; do bash test/ablation/k3s_collect.sh stock2 $sm; done

# ── Step 3: Build and deploy AIMD×3 (patched branch) ──
git checkout ablation/orig-aimd-v5
bash test/ablation/build.sh /tmp/libvgpu-patched.so
sudo cp /tmp/libvgpu-patched.so /usr/local/vgpu/libvgpu.so.v2.8.0
# verify
md5sum /tmp/libvgpu-patched.so /usr/local/vgpu/libvgpu.so.v2.8.0   # md5s must match

# ── Step 4: Collect AIMD×3 data (2 runs for error bars) ──
for sm in 0 20 40 60 80; do bash test/ablation/k3s_collect.sh origv5  $sm; done
for sm in 0 20 40 60 80; do bash test/ablation/k3s_collect.sh origv5b $sm; done

# ── Step 5: Generate comparison plot ──
python3 test/ablation/plot_final_comparison.py
# → /tmp/gpu-bench-ts/k3s_final_comparison.png

# ── Step 6: Restore vanilla after experiment ──
sudo cp /usr/local/vgpu/libvgpu.so.v2.8.0.stock /usr/local/vgpu/libvgpu.so.v2.8.0
```

> Note: `ablation/stock` branch's purpose is now mostly documentation / parity
> with vanilla. Since the bind-mount target is always `libvgpu.so.v2.8.0`,
> rebuilding from `ablation/stock` is unnecessary unless you need the
> debug-symbol binary; using the .stock backup of the vanilla file is simpler.

Data is saved to `/tmp/gpu-bench-ts/k3s/`. Each run produces two files:
- `<label>_sm<N>_gpuburn.log` — gpu_burn throughput log
- `<label>_sm<N>_smi.csv` — nvidia-smi SM utilization at 100ms intervals

### Multi-pod test

Test proportional GPU sharing between concurrent pods on the same GPU.

```bash
bash test/ablation/k3s_collect.sh test 0                          # baseline (single-pod, no limit)
bash test/ablation/k3s_multi_collect.sh mp1 a:20 b:30             # 50% total
python3 test/ablation/plot_multi_single.py mp1 a:20 b:30          # → /tmp/gpu-bench-ts/k3s/mp1_multi_plot.png
```

Each run creates all pods simultaneously and collects:
- `<label>_multi_<pod>_sm<gpucores>_gpuburn.log` — per-pod gpu_burn log
- `<label>_multi_smi.csv` — GPU-wide SM utilization (100ms intervals)
- `<label>_multi_pmon.csv` — per-process SM utilization (nvidia-smi pmon, 1s intervals)

## Scripts

| File | Purpose |
|------|---------|
| `build.sh` | Build `libvgpu.so` from current branch via Docker |
| `k3s_collect.sh` | Run one single-pod gpu_burn benchmark (30s) and collect gpu_burn log + nvidia-smi CSV |
| `k3s_multi_collect.sh` | Run multi-pod gpu_burn benchmark (30s) with per-process SM monitoring |
| `plot_single.py` | Plot a single-pod run (`python3 plot_single.py <label> <sm>`) |
| `plot_multi_single.py` | Plot a multi-pod run (`python3 plot_multi_single.py <label> <pod:cores> ...`) |
| `plot_final_comparison.py` | Generate comparison plot (expects labels: `stock`, `stock2`, `origv5`, `origv5b`) |

## Results

### RTX 4080 SUPER (original tuning target)

| Variant | sm=0 | sm=20 | sm=40 | sm=60 | sm=80 | MAE |
|---------|------|-------|-------|-------|-------|-----|
| Original (v2.8.0) | 12474±35 | 40.2% | 64.7% | 83.6% | 95.2% | **20.7%** |
| Orig + AIMD×3 | 12443±50 | 20.8% | 41.1% | 58.7% | 74.4% | **2.3%** |

### RTX 4080 (re-measured 2026-05-19, throughput basis)

| Variant | sm=0 (proc'd) | sm=20 | sm=40 | sm=60 | sm=80 | MAE |
|---------|---------------|-------|-------|-------|-------|-----|
| Original (v2.8.0) | 37184 ± 1494 | 33.7% | 64.2% | 79.2% | 89.5% | **16.6%** |
| Orig + AIMD×3 | 38097 ± 581 | 17.4% | 37.5% | 54.7% | 74.1% | **4.1%** |

### RTX 4080, SM-utilization basis (`nvidia-smi`, steady-window mean)

| Variant | sm=20 util | sm=40 util | sm=60 util | sm=80 util | MAE |
|---------|------------|------------|------------|------------|-----|
| Original (v2.8.0) | 31.4% | 59.3% | 77.3% | 88.9% | **14.2%** |
| Orig + AIMD×3 | 16.3% | 32.7% | 49.0% | 69.5% | **8.1%** |

On RTX 4080 the patch still cuts throughput-MAE by ~4× (16.6 % → 4.1 %), but
does not reach the SUPER result. AIMD now slightly under-shoots at higher `sm`,
suggesting the multiplier and AI step need GPU-specific re-tuning.

## Notes

- The benchmark scripts set `GPU_CORE_UTILIZATION_POLICY=FORCE` to bypass the
  shared memory gate. In normal HAMi deployments, the device plugin sets
  `utilization_switch` automatically when `gpucores` is configured, so `FORCE`
  is not needed for production workloads.
- The ×3, /400, /3 constants were tuned on RTX 4080 SUPER + gpu_burn.
  Different GPUs or workloads (e.g., vLLM) may require adjustment.

## Known issues / gotchas

- **Deploy target is `libvgpu.so.v2.8.0`, not `libvgpu.so`** (see warning at
  the top). All earlier revisions of this README told users to overwrite
  `libvgpu.so`, which silently has no effect under HAMi v2.8.0. Symptom: both
  Stock and AIMD sweeps yield ~identical MAE near the vanilla Stock baseline.
- **Confirm the patched code is actually running** with the probe pod recipe
  above. The two cheap signals are: zero `delta:` log lines, and `userutil1=`
  log lines referencing source line ~222.
- If you reinstall HAMi via Helm, `vgpu-init.sh` may re-copy the vanilla
  `libvgpu.so.v2.8.0` from the device-plugin image; redeploy the patch after
  any HAMi upgrade.
- Test data and the comparison plot are written to `/tmp/gpu-bench-ts/`,
  which is wiped on reboot. Copy elsewhere if you want to keep them.
