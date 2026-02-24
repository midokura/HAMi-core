#!/usr/bin/env python3
"""Plot a single multi-pod benchmark run.

Usage:
    python3 plot_multi_single.py <label> <pod1:gpucores> [pod2:gpucores] ... [--baseline N]

Examples:
    python3 plot_multi_single.py mp1 a:20 b:30
    python3 plot_multi_single.py mp1 a:20 b:30 --baseline 12443

Data pipeline:
  - gpu_burn logs: cumulative proc'd count (raw from kubectl logs)
  - nvidia-smi smi.csv: GPU-wide SM utilization at 100ms intervals (includes
    all host processes; HAMi controls per-container SM via NVML per-process API,
    so GPU-wide reading reflects the controller's effective behavior)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import csv
import sys


def parse_gpuburn_log(path):
    with open(path) as f:
        raw = f.read()
    entries = re.findall(r'(\d+\.\d+)%\s+proc\'d:\s+(\d+)\s+\((\d+)\s+Gflop/s\)', raw)
    if not entries:
        return [], 0
    results = [(float(p), int(c)) for p, c, _ in entries]
    ts_data = [(pct / 100.0 * 30.0, procd) for pct, procd in results]
    return ts_data, results[-1][1]


def parse_smi_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            try:
                rows.append((float(row[0].strip()), int(row[1].strip())))
            except (ValueError, IndexError):
                continue
    if not rows:
        return []
    t0 = rows[0][0]
    return [(t - t0, u) for t, u in rows]



def running_avg(times, values, window=2.0, min_pts=3):
    rt, rv = [], []
    for i in range(len(times)):
        t = times[i]
        mask = (times >= t - window) & (times <= t)
        idx = np.where(mask)[0]
        if len(idx) >= min_pts:
            rv.append(np.mean(values[idx]))
            rt.append(t)
    return rt, rv


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <label> <pod1:gpucores> [pod2:gpucores] ... [--baseline N]")
        print(f"Example: {sys.argv[0]} mp1 a:20 b:30")
        sys.exit(1)

    label = sys.argv[1]
    pod_specs = []
    baseline_arg = 0
    args_iter = iter(sys.argv[2:])
    for arg in args_iter:
        if arg == "--baseline":
            baseline_arg = int(next(args_iter))
        else:
            name, cores = arg.split(":")
            pod_specs.append((name, int(cores)))

    basedir = "/tmp/gpu-bench-ts/k3s"

    # Load gpu_burn logs per pod
    pod_data = {}
    for name, cores in pod_specs:
        path = f"{basedir}/{label}_multi_{name}_sm{cores}_gpuburn.log"
        ts_data, final = parse_gpuburn_log(path)
        pod_data[name] = {"cores": cores, "ts": ts_data, "final": final}

    # Load GPU-wide SM util
    smi = parse_smi_csv(f"{basedir}/{label}_multi_smi.csv")

    # Load baseline (single-pod sm=0)
    baseline = baseline_arg
    baseline_ts = []
    if baseline == 0:
        # Try auto-detect from common labels
        for try_label in [label, "test", "origv5", "stock"]:
            try:
                baseline_ts, baseline = parse_gpuburn_log(
                    f"{basedir}/{try_label}_sm0_gpuburn.log")
                print(f"Baseline (from {try_label}_sm0): {baseline} proc'd")
                break
            except FileNotFoundError:
                continue
    if baseline == 0:
        print("No baseline found. Use --baseline N for target lines.")

    print(f"Run: {label}")
    total_target = sum(cores for _, cores in pod_specs)
    for name, cores in pod_specs:
        d = pod_data[name]
        pct_str = f" ({d['final']/baseline*100:.1f}%)" if baseline > 0 else ""
        print(f"  Pod {name} (gpucores={cores}): {d['final']} proc'd{pct_str}")
    print(f"  Total target: {total_target}%")

    # Colors per pod
    pod_colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    pod_colors = {}
    for i, (name, _) in enumerate(pod_specs):
        pod_colors[name] = pod_colors_list[i % len(pod_colors_list)]

    # Plot: 2 panels
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    specs_str = " + ".join(f"{name}:{cores}%" for name, cores in pod_specs)
    fig.suptitle(f"Multi-pod benchmark: {specs_str}\n"
                 f"label={label}, gpu_burn 30s, k3s",
                 fontsize=13, fontweight='bold')

    # --- Panel 1: GPU-wide SM Utilization (nvidia-smi) ---
    ax = axes[0]
    ax.set_title("nvidia-smi SM Utilization (GPU-wide)")
    if smi:
        smi_arr = np.array(smi)
        mask30 = smi_arr[:, 0] <= 30.0
        times, utils = smi_arr[mask30, 0], smi_arr[mask30, 1]
        ax.plot(times, utils, color='#999999', alpha=0.5, linewidth=1.5, label='Raw (100ms)')
        rt, rv = running_avg(times, utils)
        if rt:
            avg_val = np.mean(utils[int(len(utils)*0.2):])
            ax.plot(rt, rv, color='#ff7f0e', linewidth=2,
                    label=f'Running avg 2s (steady: {avg_val:.0f}%)')
        ax.axhline(total_target, color='#2ca02c', linestyle='--', linewidth=2,
                   label=f'Total target ({total_target}%)')
    ax.set_ylabel("SM Util (%)")
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Cumulative Throughput ---
    ax = axes[1]
    ax.set_title("Cumulative Throughput (gpu_burn proc'd)")

    # Baseline curve (sm=0, gray)
    if baseline > 0 and len(baseline_ts) >= 2:
        barr = np.array(baseline_ts)
        ax.plot(barr[:, 0], barr[:, 1], color='#999999', linewidth=1.5,
                label=f'baseline sm=0 ({baseline})')

    for name, cores in pod_specs:
        d = pod_data[name]
        ts = d["ts"]
        pct_str = f", {d['final']/baseline*100:.0f}%" if baseline > 0 else ""
        if len(ts) >= 2:
            arr = np.array(ts)
            ax.plot(arr[:, 0], arr[:, 1], color=pod_colors[name], linewidth=2,
                    label=f'{name} gpucores={cores} ({d["final"]}{pct_str})')

        # Target line: baseline × gpucores%
        if baseline > 0 and len(baseline_ts) >= 2:
            ax.plot(barr[:, 0], barr[:, 1] * cores / 100,
                    color=pod_colors[name], linestyle='--', linewidth=1, alpha=0.5,
                    label=f'target {cores}%')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("proc'd (cumulative)")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = f"{basedir}/{label}_multi_plot.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
