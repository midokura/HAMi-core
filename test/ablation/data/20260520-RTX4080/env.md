# Environment for 20260520-RTX4080

Measurement run on host `phoenix`, 2026-05-20.

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 |
| Driver | 580.126.20 |
| VBIOS | 95.03.2B.40.02 |
| CUDA | 13.0 (V13.0.88) |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.17.0-23-generic |
| k3s | v1.33.6+k3s1 |
| HAMi | 2.8.0 (helm chart) |
| HAMi-core branch | `ablation/orig-aimd-v5` (commit 7035f5a) |
| Deploy target | `/usr/local/vgpu/libvgpu.so.v2.8.0` |
| Patched binary md5 | `9166d0e81507fcba8aa5686603e3e737` |
| Vanilla v2.8.0 md5 | `a15ee4baa7ba4d6623a2ab0911d21603` |
| Workload | `oguzpastirmaci/gpu-burn`, 30 s per run |
| Sweeps | `stock`, `stock2`, `origv5`, `origv5b` × sm ∈ {0,20,40,60,80} |

## Results (`plot_final_comparison.py`)

Throughput basis:

| Variant | sm=0 (proc'd) | sm=20 | sm=40 | sm=60 | sm=80 | MAE |
|---|---|---|---|---|---|---|
| Original (v2.8.0) | 38263 ± 415  | 36.9% | 64.7% | 80.2% | 88.1% | **17.5%** |
| Orig + AIMD×3     | 37682 ± 1328 | 18.8% | 38.9% | 57.7% | 73.2% | **2.8%**  |

SM-utilization basis (steady-window mean, `nvidia-smi`):

| Variant | sm=20 | sm=40 | sm=60 | sm=80 | MAE |
|---|---|---|---|---|---|
| Original (v2.8.0) | 34.8% | 60.5% | 81.9% | 87.1% | **16.1%** |
| Orig + AIMD×3     | 18.2% | 35.1% | 53.3% | 70.0% | **5.9%**  |

This re-measurement was performed from a fresh `git clone` to validate the
README procedure end-to-end after the `.stock` backup convention and the
`libvgpu.so.v2.8.0` deploy-target fix.
