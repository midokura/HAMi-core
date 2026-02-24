#!/usr/bin/env python3
"""Plot a single k3s_collect.sh run: SM utilization + throughput time series.

Usage:
    python3 plot_single.py <label> <sm> [--baseline N] [--basedir DIR]

Examples:
    python3 plot_single.py test 40
    python3 plot_single.py origv5 60 --baseline 12443
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
import re
import argparse


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


def running_avg(times, values, window=2.0, min_pts=5):
    rt, rv = [], []
    for i in range(len(times)):
        t = times[i]
        mask = (times >= t - window) & (times <= t)
        idx = np.where(mask)[0]
        if len(idx) >= min_pts:
            rv.append(np.mean(values[idx]))
            rt.append(t)
    return rt, rv


def steady_state_rate(ts_data, skip_fraction=0.2):
    """Compute steady-state throughput rate, excluding initial ramp-up."""
    if len(ts_data) < 2:
        return 0
    arr = np.array(ts_data)
    times, counts = arr[:, 0], arr[:, 1]
    total_time = times[-1] - times[0]
    cutoff = times[0] + total_time * skip_fraction
    mask = times >= cutoff
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return counts[-1] / times[-1] if times[-1] > 0 else 0
    dt = times[idx[-1]] - times[idx[0]]
    dc = counts[idx[-1]] - counts[idx[0]]
    return dc / dt if dt > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Plot single k3s_collect.sh run")
    parser.add_argument("label", help="Run label (e.g., test, origv5)")
    parser.add_argument("sm", type=int, help="gpucores target (e.g., 40)")
    parser.add_argument("--baseline", type=int, default=0,
                        help="Baseline proc'd count (from sm=0 run). "
                             "Auto-detected from <label>_sm0 if available.")
    parser.add_argument("--basedir", default="/tmp/gpu-bench-ts/k3s",
                        help="Data directory (default: /tmp/gpu-bench-ts/k3s)")
    args = parser.parse_args()

    gpuburn_path = f"{args.basedir}/{args.label}_sm{args.sm}_gpuburn.log"
    smi_path = f"{args.basedir}/{args.label}_sm{args.sm}_smi.csv"

    # Parse data
    ts_data, final_procd = parse_gpuburn_log(gpuburn_path)
    smi = parse_smi_csv(smi_path)

    # Auto-detect baseline
    baseline = args.baseline
    baseline_ts = []
    if baseline == 0:
        try:
            baseline_ts, baseline = parse_gpuburn_log(
                f"{args.basedir}/{args.label}_sm0_gpuburn.log")
            print(f"Baseline (from {args.label}_sm0): {baseline} proc'd")
        except FileNotFoundError:
            pass
    if baseline == 0:
        baseline = final_procd
        print(f"No baseline found — using this run's final count as reference: {baseline}")
        has_baseline = False
    else:
        has_baseline = True

    actual_pct = final_procd / baseline * 100 if baseline > 0 else 0
    print(f"Run: {args.label} sm={args.sm}")
    print(f"Final: {final_procd} proc'd ({actual_pct:.1f}% of baseline)")
    if args.sm > 0 and has_baseline:
        print(f"Error: {actual_pct - args.sm:+.1f}pp")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"gpucores={args.sm}  label={args.label}  "
                 f"({final_procd} proc'd, {actual_pct:.1f}%)\n"
                 f"gpu_burn 30s, RTX 4080 SUPER, k3s",
                 fontsize=13, fontweight='bold')

    # --- Panel 1: nvidia-smi SM Utilization ---
    ax = axes[0]
    ax.set_title("nvidia-smi SM Utilization")
    if smi:
        smi_arr = np.array(smi)
        mask30 = smi_arr[:, 0] <= 30.0
        times, utils = smi_arr[mask30, 0], smi_arr[mask30, 1]
        ax.plot(times, utils, color='#999999', alpha=0.6, linewidth=1.5, label='Raw (100ms)')
        rt, rv = running_avg(times, utils)
        if rt:
            avg_val = np.mean(utils[int(len(utils)*0.2):])
            ax.plot(rt, rv, color='#ff7f0e', linewidth=2,
                    label=f'Running avg 2s (steady: {avg_val:.0f}%)')
    if args.sm > 0:
        ax.axhline(args.sm, color='#2ca02c', linestyle='--', linewidth=2,
                    label=f'Target ({args.sm}%)')
    ax.set_ylabel("SM Util (%)")
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Throughput (cumulative proc'd) ---
    ax = axes[1]
    if has_baseline and baseline_ts:
        ax.set_title(f"Cumulative Throughput — {actual_pct:.1f}% of baseline "
                     f"(target {args.sm}%)")

        # Baseline (sm=0) curve
        barr = np.array(baseline_ts)
        ax.plot(barr[:, 0], barr[:, 1], color='#999999', linewidth=2,
                label=f'Baseline sm=0 ({baseline} proc\'d)')

        # Ideal target line: baseline × target%
        if args.sm > 0:
            ax.plot(barr[:, 0], barr[:, 1] * args.sm / 100,
                    color='#2ca02c', linestyle='--', linewidth=2,
                    label=f'Ideal {args.sm}% ({baseline * args.sm // 100} proc\'d)')

        # Test curve
        if len(ts_data) >= 2:
            tarr = np.array(ts_data)
            ax.plot(tarr[:, 0], tarr[:, 1], color='#1f77b4', linewidth=2.5,
                    label=f'sm={args.sm} ({final_procd} proc\'d)')

        ax.set_ylabel("proc'd (cumulative)")
        ax.legend(loc='upper left', fontsize=9)
    elif has_baseline:
        ax.set_title(f"Throughput — {actual_pct:.1f}% of baseline (target {args.sm}%)")
        if len(ts_data) >= 2:
            arr = np.array(ts_data)
            ax.plot(arr[:, 0], arr[:, 1], color='#1f77b4', linewidth=2,
                    label=f'sm={args.sm} ({final_procd} proc\'d)')
        ax.set_ylabel("proc'd (cumulative)")
        ax.legend(loc='upper left', fontsize=9)
    else:
        ax.set_title("Throughput (raw proc'd count)")
        if len(ts_data) >= 2:
            arr = np.array(ts_data)
            ax.plot(arr[:, 0], arr[:, 1], color='#1f77b4', linewidth=2,
                    label=f'sm={args.sm} ({final_procd} proc\'d)')
        ax.set_ylabel("proc'd (cumulative)")
        ax.legend(loc='upper left', fontsize=9)

    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = f"{args.basedir}/{args.label}_sm{args.sm}_plot.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
