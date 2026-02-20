# Ablation Study: gpucores Controller Variants

Compares 6 controller variants to isolate the contribution of each change.

## Branches and Variants

| Branch | Algorithm | ×3 | NVML fixes | File label | Description |
|--------|-----------|-----|------------|------------|-------------|
| `ablation/stock-delta` | Stock delta() | No | Yes | `sd` | Stock + NVML fixes only |
| `ablation/aimd` | AIMD | No | Yes | `aimd` | AIMD without ×3 |
| `ablation/stock-delta-x3` | Stock delta() | Yes | Yes | `sdx3` | Stock + ×3 + NVML fixes |
| `ablation/orig-aimd-v5` | AIMD | Yes | **No** | `origv5` | Original + AIMD×3 (minimal patch) |
| `ablation/aimd-v5` | AIMD | Yes | Yes | `v5` | Full v5 (AIMD + ×3 + NVML) |

Additionally, the upstream HAMi v2.8.0 binary (`stock` label) is used as the baseline.

## Results (RTX 4080 SUPER, k3s, gpu_burn 30s)

| Variant | MAE | sm=20 | sm=40 | sm=60 | sm=80 |
|---------|-----|-------|-------|-------|-------|
| Original (v2.8.0) | 20.9% | 40.2% | 64.7% | 83.6% | 95.2% |
| Stock + NVML fixes | 23.9% | 50.8% | 71.6% | 83.6% | 89.6% |
| AIMD (no ×3) | 33.1% | 6.5% | 13.4% | 20.3% | 27.3% |
| Stock + ×3 | 21.9% | 44.3% | 69.3% | 83.1% | 91.0% |
| **Orig + AIMD×3 (no NVML)** | **2.2%** | 20.8% | 41.1% | 58.7% | 74.4% |
| **AIMD v5 (×3+NVML)** | **0.9%** | 20.8% | 40.2% | 60.5% | 82.2% |

### Key Findings

1. **AIMD + ×3 is the essential fix** — both components required together
2. **NVML fixes are not essential** — Original + AIMD×3 achieves MAE 2.2% without them
3. NVML fixes improve sm=80 accuracy (74.4% → 82.2%), reducing MAE from 2.2% to 0.9%
4. Neither AIMD alone (MAE 33.1%) nor ×3 alone (MAE 21.9%) improves accuracy
5. NVML fixes actually **worsen** stock delta (20.9% → 23.9%)

### Recommended Minimal Patch

For upstreaming, `ablation/orig-aimd-v5` is the minimal change:
- **Only modifies `utilization_watcher()` body** (15 lines changed)
- No changes to `get_used_gpu_utilization()` or `delta()`
- MAE 2.2% vs Original's 20.9%

## Key Differences

### Control Algorithm
- **Stock delta()**: Additive token replenishment via `change_token(share)` which adds to `g_cur_cuda_cores`. Symmetric increment/decrement based on utilization gap.
- **AIMD**: Direct assignment `g_cur_cuda_cores = share`. Additive increase with proportional step, multiplicative decrease (÷3 on overshoot). Effective limit = 7/8 of target.

### ×3 Scaling
- **Without ×3**: `base = SM_NUM × MAX_THREAD_PER_SM` (116,736 on RTX 4080)
- **With ×3**: `base = SM_NUM × MAX_THREAD_PER_SM × 3` (350,208 on RTX 4080)

The ×3 factor compensates for the mismatch between token units and actual GPU throughput units.

### NVML Fixes (in full v5 only)
- `lastSeenTimeStamp=0`: Retrieve all buffered NVML samples
- Decay stale utilization: Halve `last_userutil` when no fresh sample
- `share_floor`: Dynamic floor in `delta()` (unused by AIMD path)
- Device index fix: Use NVML index instead of CUDA index

## Generated Plots

`plot_ablation.py` generates three plots:

| File | Description |
|------|-------------|
| `k3s_ablation.png` | Main comparison: sweep accuracy, error bars, MAE ranking, data table |
| `k3s_ablation_ts_sm40.png` | 6-panel time-series at gpucores=40: throughput, SM util, cumulative, bars |
| `k3s_ablation_multi.png` | Multi-level comparison: all variants × all gpucores levels |

## Running the Benchmark

### Prerequisites
- k3s cluster with HAMi device plugin installed
- `gpu_burn` container image available
- `nvidia-smi` accessible on the host

### Quick Test (single level)

```bash
# Build variant binary (from libvgpu root)
./test/ablation/build_variants.sh

# Deploy a variant
sudo cp /tmp/libvgpu-<variant>.so /usr/local/vgpu/libvgpu.so

# Run single benchmark at gpucores=40
bash test/ablation/k3s_collect.sh v5 40
```

### Full Sweep

```bash
# Run sweep across multiple gpucores levels
bash test/ablation/k3s_sweep.sh v5 "0 20 40 60 80"
```

### Automated Ablation (all variants)

```bash
# Build all variants
bash test/ablation/build_variants.sh

# Run full ablation (requires manual binary deployment between variants)
bash test/ablation/run_ablation.sh
```

### Generate Comparison Plot

```bash
python3 test/ablation/plot_ablation.py
# Output: /tmp/gpu-bench-ts/k3s_ablation.png
```

## Data Format

### gpu_burn log
Each line contains a timestamp and throughput count:
```
14:23:45 - 1234 proc'd
```

### nvidia-smi CSV
```csv
timestamp,utilization_gpu
1708123456.789,45
```
