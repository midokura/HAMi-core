#!/usr/bin/env python3
"""
Ablation study comparison plot for HAMi gpucores controller variants.

Generates:
  /tmp/gpu-bench-ts/k3s_ablation.png         - Main accuracy comparison
  /tmp/gpu-bench-ts/k3s_ablation_ts_sm40.png  - Time-series detail at sm=40

Usage:
  python3 plot_ablation.py [--datadir /tmp/gpu-bench-ts/k3s]
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import csv
import os
import argparse

# ============================================================
# Parsers (shared with plot_k3s_comprehensive.py)
# ============================================================

def parse_gpuburn_log(path):
    """Parse gpu_burn log. Returns list of (elapsed_s, proc'd), final_procd."""
    with open(path) as f:
        raw = f.read()
    entries = re.findall(r'(\d+\.\d+)%\s+proc\'d:\s+(\d+)\s+\((\d+)\s+Gflop/s\)', raw)
    if not entries:
        return [], 0
    results = []
    for pct_str, procd_str, _ in entries:
        pct = float(pct_str)
        procd = int(procd_str)
        results.append((pct, procd))
    duration = 30.0
    ts_data = [(pct / 100.0 * duration, procd) for pct, procd in results]
    return ts_data, results[-1][1]


def parse_smi_csv(path):
    """Parse nvidia-smi CSV. Returns list of (elapsed_s, sm_util%)."""
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


def calc_instantaneous_util(ts_data, baseline_final, window=1.0):
    """Compute instantaneous throughput (% of baseline) via sliding window."""
    if len(ts_data) < 2:
        return [], []
    arr = np.array(ts_data)
    times, counts = arr[:, 0], arr[:, 1]
    baseline_rate = baseline_final / 30.0
    rt, rv = [], []
    for i in range(len(times)):
        t = times[i]
        mask = (times >= t - window) & (times <= t)
        idx = np.where(mask)[0]
        if len(idx) >= 2:
            dt = times[idx[-1]] - times[idx[0]]
            dc = counts[idx[-1]] - counts[idx[0]]
            if dt > 0.1:
                util_pct = (dc / dt) / baseline_rate * 100
                rt.append(t)
                rv.append(util_pct)
    return rt, rv


def get_stall_intervals(ts_data, min_stall_count=3):
    """Return list of (start_t, end_t) for OFF (stall) intervals."""
    if len(ts_data) < 2:
        return []
    intervals = []
    prev_procd = ts_data[0][1]
    stall_start = None
    stall_count = 0
    for i in range(1, len(ts_data)):
        t, procd = ts_data[i]
        if procd == prev_procd:
            if stall_count == 0:
                stall_start = ts_data[i-1][0]
            stall_count += 1
        else:
            if stall_count >= min_stall_count and stall_start is not None:
                intervals.append((stall_start, t))
            stall_count = 0
            stall_start = None
        prev_procd = procd
    if stall_count >= min_stall_count and stall_start is not None:
        intervals.append((stall_start, ts_data[-1][0]))
    return intervals


def plot_stall_overlay(ax, intervals, color='gray', alpha=0.2, label=None):
    """Overlay shaded bands for OFF (stall) intervals."""
    for i, (t0, t1) in enumerate(intervals):
        ax.axvspan(t0, t1, alpha=alpha, color=color,
                   label=label if i == 0 else None)


# ============================================================
# Configuration
# ============================================================

# Variant definitions: (label_in_files, display_name, color, linestyle)
VARIANTS = [
    ("stock",  "Original (v2.8.0)",       "#2ca02c", "-"),   # green — upstream HAMi
    ("sd",     "Stock + NVML fixes",      "#d62728", "-"),   # red
    ("aimd",   "AIMD (no ×3)",            "#ff7f0e", "--"),  # orange
    ("sdx3",   "Stock + ×3",             "#9467bd", "-."),  # purple
    ("origv5", "Orig + AIMD×3 (no NVML)", "#17becf", "-"),   # cyan
    ("v5",     "AIMD v5 (×3+NVML)",       "#1f77b4", "-"),   # blue — full v5
]

LEVELS = [0, 20, 40, 60, 80]


def load_data(basedir):
    """Load all variant data from basedir."""
    data = {}
    for file_label, display_name, color, ls in VARIANTS:
        data[file_label] = {"name": display_name, "color": color, "ls": ls, "levels": {}}
        for sm in LEVELS:
            gpulog = f"{basedir}/{file_label}_sm{sm}_gpuburn.log"
            smilog = f"{basedir}/{file_label}_sm{sm}_smi.csv"
            if os.path.exists(gpulog):
                ts, final = parse_gpuburn_log(gpulog)
                smi = parse_smi_csv(smilog) if os.path.exists(smilog) else []
                data[file_label]["levels"][sm] = {"ts": ts, "final": final, "smi": smi}
    return data


def find_baseline(data):
    """Find baseline (no-limit) throughput from any variant's sm=0."""
    baselines = []
    for v in data.values():
        if 0 in v["levels"]:
            baselines.append(v["levels"][0]["final"])
    if not baselines:
        raise ValueError("No baseline (sm=0) data found")
    return int(np.mean(baselines))


# ============================================================
# PLOT 1: Accuracy comparison (main ablation plot)
# ============================================================

def plot_accuracy(data, baseline_final, outpath):
    """2x2 plot: sweep accuracy + error bars + MAE comparison + detailed table."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('HAMi gpucores Ablation Study\n'
                 'k3s, gpu_burn 30s, RTX 4080 SUPER',
                 fontsize=14, fontweight='bold')

    # --- Panel 1: Sweep accuracy (actual vs target) ---
    ax = axes[0, 0]
    ax.set_title("Throughput vs Target")
    # Ideal line
    targets = [sm for sm in LEVELS if sm > 0]
    ax.plot(targets, targets, 'k--', alpha=0.5, label='Ideal', linewidth=2)

    for file_label, display_name, color, ls in VARIANTS:
        v = data[file_label]
        xs, ys = [], []
        for sm in LEVELS:
            if sm == 0 or sm not in v["levels"]:
                continue
            actual_pct = v["levels"][sm]["final"] / baseline_final * 100
            xs.append(sm)
            ys.append(actual_pct)
        if xs:
            ax.plot(xs, ys, color=color, linestyle=ls, marker='o',
                    linewidth=2, markersize=8, label=display_name)

    ax.set_xlabel("gpucores Target (%)")
    ax.set_ylabel("Actual Throughput (% of baseline)")
    ax.legend(loc='upper left')
    ax.set_xlim(10, 90)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Error (actual - target) ---
    ax = axes[0, 1]
    ax.set_title("Error (Actual − Target)")
    ax.axhline(0, color='k', linewidth=1, alpha=0.5)

    for file_label, display_name, color, ls in VARIANTS:
        v = data[file_label]
        xs, errs = [], []
        for sm in LEVELS:
            if sm == 0 or sm not in v["levels"]:
                continue
            actual_pct = v["levels"][sm]["final"] / baseline_final * 100
            xs.append(sm)
            errs.append(actual_pct - sm)
        if xs:
            n = len(VARIANTS)
            idx = VARIANTS.index((file_label, display_name, color, ls))
            w = 16.0 / n  # bar width scales with variant count
            offset = (idx - (n - 1) / 2) * w
            ax.bar([x + offset for x in xs],
                   errs, width=w * 0.9, color=color, alpha=0.7, label=display_name)

    ax.set_xlabel("gpucores Target (%)")
    ax.set_ylabel("Error (percentage points)")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 3: MAE summary ---
    ax = axes[1, 0]
    ax.set_title("Mean Absolute Error (MAE)")

    mae_data = []
    for file_label, display_name, color, ls in VARIANTS:
        v = data[file_label]
        errors = []
        for sm in LEVELS:
            if sm == 0 or sm not in v["levels"]:
                continue
            actual_pct = v["levels"][sm]["final"] / baseline_final * 100
            errors.append(abs(actual_pct - sm))
        if errors:
            mae = np.mean(errors)
            mae_data.append((display_name, mae, color))

    if mae_data:
        names, maes, colors = zip(*mae_data)
        bars = ax.barh(range(len(names)), maes, color=colors, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("MAE (percentage points)")
        for bar, mae in zip(bars, maes):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{mae:.1f}%', va='center', fontsize=11, fontweight='bold')
        ax.set_xlim(0, max(maes) * 1.3)
        ax.grid(True, alpha=0.3, axis='x')

    # --- Panel 4: Data table ---
    ax = axes[1, 1]
    ax.set_title("Raw Data")
    ax.axis('off')

    headers = ["Variant", "sm=0\n(base)"] + [f"sm={sm}\n(target)" for sm in LEVELS if sm > 0] + ["MAE"]
    table_data = []
    for file_label, display_name, color, ls in VARIANTS:
        v = data[file_label]
        row = [display_name]
        errors = []
        for sm in LEVELS:
            if sm not in v["levels"]:
                row.append("—")
                continue
            final = v["levels"][sm]["final"]
            pct = final / baseline_final * 100
            if sm == 0:
                row.append(f"{final}")
            else:
                row.append(f"{pct:.1f}%")
                errors.append(abs(pct - sm))
        mae = np.mean(errors) if errors else float('nan')
        row.append(f"{mae:.1f}%")
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Color the MAE column
    for i, (file_label, display_name, color, ls) in enumerate(VARIANTS):
        table[i + 1, 0].set_text_props(color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


# ============================================================
# PLOT 2: Time-series detail at sm=40
# ============================================================

def plot_timeseries_sm40(data, baseline_final, outpath):
    """6-panel time-series comparison at gpucores=40 for all variants."""
    sm = 40
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle(f'HAMi gpucores={sm} Time-Series: All Variants\n'
                 f'k3s, gpu_burn 30s, RTX 4080 SUPER',
                 fontsize=14, fontweight='bold')

    # Panel 1: Instantaneous throughput
    ax = axes[0, 0]
    ax.set_title("Instantaneous Throughput (1s window)")
    for file_label, display_name, color, ls in VARIANTS:
        v = data[file_label]
        if sm not in v["levels"]:
            continue
        d = v["levels"][sm]
        rt, rv = calc_instantaneous_util(d["ts"], baseline_final)
        if rt:
            ax.plot(rt, rv, color=color, alpha=0.8, linewidth=1, label=display_name)
            stalls = get_stall_intervals(d["ts"])
            plot_stall_overlay(ax, stalls, color=color, alpha=0.08)
    ax.axhline(sm, color='k', linestyle='--', alpha=0.5, label=f'Target {sm}%')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput (% of baseline)")
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 120)
    ax.grid(True, alpha=0.3)

    # Panel 2: nvidia-smi SM utilization
    ax = axes[0, 1]
    ax.set_title("nvidia-smi SM Utilization")
    for file_label, display_name, color, ls in VARIANTS:
        v = data[file_label]
        if sm not in v["levels"]:
            continue
        d = v["levels"][sm]
        if d["smi"]:
            smi_arr = np.array(d["smi"])
            ax.plot(smi_arr[:, 0], smi_arr[:, 1], color=color, alpha=0.6,
                    linewidth=0.8, label=display_name)
    ax.axhline(sm, color='k', linestyle='--', alpha=0.5, label=f'Target {sm}%')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SM Utilization (%)")
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    # Panel 3: Cumulative throughput
    ax = axes[1, 0]
    ax.set_title("Cumulative Throughput")
    for file_label, display_name, color, ls in VARIANTS:
        v = data[file_label]
        if sm not in v["levels"]:
            continue
        d = v["levels"][sm]
        if d["ts"]:
            arr = np.array(d["ts"])
            cum_pct = arr[:, 1] / baseline_final * 100
            ax.plot(arr[:, 0], cum_pct, color=color, linewidth=2,
                    linestyle=ls, label=f'{display_name} ({d["final"]/baseline_final*100:.1f}%)')
    ax.axhline(sm, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative (% of baseline)")
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Cumulative SM utilization (running average)
    ax = axes[1, 1]
    ax.set_title("Cumulative SM Utilization (running avg)")
    for file_label, display_name, color, ls in VARIANTS:
        v = data[file_label]
        if sm not in v["levels"]:
            continue
        d = v["levels"][sm]
        if d["smi"]:
            smi_arr = np.array(d["smi"])
            times = smi_arr[:, 0]
            utils = smi_arr[:, 1]
            cum_avg = np.cumsum(utils) / np.arange(1, len(utils) + 1)
            final_avg = cum_avg[-1] if len(cum_avg) > 0 else 0
            ax.plot(times, cum_avg, color=color, linewidth=2,
                    linestyle=ls, label=f'{display_name} ({final_avg:.1f}%)')
    ax.axhline(sm, color='k', linestyle='--', alpha=0.5, label=f'Target {sm}%')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SM Utilization (% running avg)")
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    # Panel 5: Final throughput bar chart
    ax = axes[2, 0]
    ax.set_title(f"Final Throughput at gpucores={sm}")
    bars_data = []
    for file_label, display_name, color, ls in VARIANTS:
        v = data[file_label]
        if sm in v["levels"]:
            pct = v["levels"][sm]["final"] / baseline_final * 100
            bars_data.append((display_name, pct, color))

    if bars_data:
        names, pcts, colors = zip(*bars_data)
        bars = ax.bar(range(len(names)), pcts, color=colors, alpha=0.8)
        ax.axhline(sm, color='k', linestyle='--', alpha=0.5, label=f'Target {sm}%')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylabel("Throughput (% of baseline)")
        for bar, pct in zip(bars, pcts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.1f}%', ha='center', fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(pcts) * 1.2)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # Panel 6: Final SM utilization bar chart
    ax = axes[2, 1]
    ax.set_title(f"Final SM Utilization (avg) at gpucores={sm}")
    bars_data = []
    for file_label, display_name, color, ls in VARIANTS:
        v = data[file_label]
        if sm in v["levels"] and v["levels"][sm]["smi"]:
            smi_arr = np.array(v["levels"][sm]["smi"])
            avg_smi = np.mean(smi_arr[:, 1])
            bars_data.append((display_name, avg_smi, color))

    if bars_data:
        names, avgs, colors = zip(*bars_data)
        bars = ax.bar(range(len(names)), avgs, color=colors, alpha=0.8)
        ax.axhline(sm, color='k', linestyle='--', alpha=0.5, label=f'Target {sm}%')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylabel("SM Utilization (% avg)")
        for bar, avg in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{avg:.1f}%', ha='center', fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(avgs) * 1.3 if max(avgs) > 0 else 100)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


# ============================================================
# PLOT 3: Multi-level time-series comparison
# ============================================================

def plot_multi_level(data, baseline_final, outpath):
    """Side-by-side time-series for each gpucores level."""
    active_levels = [sm for sm in LEVELS if sm > 0]
    n_levels = len(active_levels)

    fig, axes = plt.subplots(n_levels, 2, figsize=(18, 5 * n_levels))
    fig.suptitle('HAMi gpucores: All Variants × All Levels\n'
                 'k3s, gpu_burn 30s, RTX 4080 SUPER',
                 fontsize=14, fontweight='bold')

    for row, sm in enumerate(active_levels):
        # Left: instantaneous throughput
        ax = axes[row, 0]
        ax.set_title(f"gpucores={sm}: Instantaneous Throughput (1s)")
        for file_label, display_name, color, ls in VARIANTS:
            v = data[file_label]
            if sm not in v["levels"]:
                continue
            d = v["levels"][sm]
            rt, rv = calc_instantaneous_util(d["ts"], baseline_final)
            if rt:
                ax.plot(rt, rv, color=color, alpha=0.8, linewidth=1, label=display_name)
                stalls = get_stall_intervals(d["ts"])
                plot_stall_overlay(ax, stalls, color=color, alpha=0.08)
        ax.axhline(sm, color='k', linestyle='--', alpha=0.5, label=f'Target')
        ax.set_ylabel("Throughput (%)")
        ax.set_ylim(0, 120)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(loc='upper right', fontsize=8)
        if row == n_levels - 1:
            ax.set_xlabel("Time (s)")

        # Right: nvidia-smi
        ax = axes[row, 1]
        ax.set_title(f"gpucores={sm}: nvidia-smi SM Util")
        for file_label, display_name, color, ls in VARIANTS:
            v = data[file_label]
            if sm not in v["levels"]:
                continue
            d = v["levels"][sm]
            if d["smi"]:
                smi_arr = np.array(d["smi"])
                ax.plot(smi_arr[:, 0], smi_arr[:, 1], color=color, alpha=0.6,
                        linewidth=0.8, label=display_name)
        ax.axhline(sm, color='k', linestyle='--', alpha=0.5)
        ax.set_ylabel("SM Util (%)")
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(loc='upper right', fontsize=8)
        if row == n_levels - 1:
            ax.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study plots")
    parser.add_argument("--datadir", default="/tmp/gpu-bench-ts/k3s",
                        help="Directory containing benchmark data")
    args = parser.parse_args()

    data = load_data(args.datadir)
    baseline_final = find_baseline(data)
    print(f"Baseline: {baseline_final} proc'd ({baseline_final/30:.0f}/s)")

    # Print summary
    for file_label, display_name, color, ls in VARIANTS:
        v = data[file_label]
        errors = []
        print(f"\n{display_name}:")
        for sm in LEVELS:
            if sm not in v["levels"]:
                print(f"  sm={sm}: NO DATA")
                continue
            final = v["levels"][sm]["final"]
            pct = final / baseline_final * 100
            if sm > 0:
                errors.append(abs(pct - sm))
            print(f"  sm={sm}: {final} proc'd ({pct:.1f}%)" +
                  (f"  err={pct-sm:+.1f}pp" if sm > 0 else " (baseline)"))
        if errors:
            print(f"  MAE: {np.mean(errors):.1f}%")

    outdir = os.path.dirname(args.datadir) or args.datadir
    plot_accuracy(data, baseline_final, f"{outdir}/k3s_ablation.png")
    plot_timeseries_sm40(data, baseline_final, f"{outdir}/k3s_ablation_ts_sm40.png")
    plot_multi_level(data, baseline_final, f"{outdir}/k3s_ablation_multi.png")
