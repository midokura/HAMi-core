#!/usr/bin/env python3
"""Final comparison: Original vs Orig + AIMD×3 (2 runs each).

Data pipeline:
  - gpu_burn logs: cumulative proc'd count at each time point (raw from kubectl logs)
  - Throughput = final proc'd / baseline proc'd × 100 (no windowing or smoothing)
  - Baseline = mean of all sm=0 runs across both variants
  - Cumulative throughput panels plot raw proc'd counters directly
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import csv


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


def parse_gpuburn_log(path):
    with open(path) as f:
        raw = f.read()
    entries = re.findall(r'(\d+\.\d+)%\s+proc\'d:\s+(\d+)\s+\((\d+)\s+Gflop/s\)', raw)
    if not entries:
        return [], 0
    results = [(float(p), int(c)) for p, c, _ in entries]
    ts_data = [(pct / 100.0 * 30.0, procd) for pct, procd in results]
    return ts_data, results[-1][1]


# ============================================================
# Load data: 2 runs each
# ============================================================
basedir = "/tmp/gpu-bench-ts/k3s"
levels = [0, 20, 40, 60, 80]
targets = [sm for sm in levels if sm > 0]

runs = {
    "Original (v2.8.0)": [
        {sm: parse_gpuburn_log(f"{basedir}/stock_sm{sm}_gpuburn.log") for sm in levels},
        {sm: parse_gpuburn_log(f"{basedir}/stock2_sm{sm}_gpuburn.log") for sm in levels},
    ],
    "Orig + AIMD×3": [
        {sm: parse_gpuburn_log(f"{basedir}/origv5_sm{sm}_gpuburn.log") for sm in levels},
        {sm: parse_gpuburn_log(f"{basedir}/origv5b_sm{sm}_gpuburn.log") for sm in levels},
    ],
}

smi_runs = {
    "Original (v2.8.0)": {
        sm: parse_smi_csv(f"{basedir}/stock2_sm{sm}_smi.csv") for sm in targets
    },
    "Orig + AIMD×3": {
        sm: parse_smi_csv(f"{basedir}/origv5b_sm{sm}_smi.csv") for sm in targets
    },
}

variant_colors = {"Original (v2.8.0)": "#d62728", "Orig + AIMD×3": "#1f77b4"}
level_colors = {20: "#e377c2", 40: "#ff7f0e", 60: "#2ca02c", 80: "#9467bd"}

# Compute per-variant stats
stats = {}
for name, run_list in runs.items():
    stats[name] = {}
    for sm in levels:
        finals = [run[sm][1] for run in run_list]
        stats[name][sm] = {"finals": finals, "mean": np.mean(finals), "std": np.std(finals)}

# Use mean of all sm=0 as baseline
all_baselines = []
for name in runs:
    all_baselines.extend(stats[name][0]["finals"])
baseline = np.mean(all_baselines)

print(f"Baseline (avg of all sm=0 runs): {baseline:.0f} proc'd")
for name in runs:
    errs = []
    for sm in levels:
        pct = stats[name][sm]["mean"] / baseline * 100
        if sm > 0:
            errs.append(abs(pct - sm))
        print(f"  {name} sm={sm}: {stats[name][sm]['mean']:.0f} ({pct:.1f}%) "
              f"± {stats[name][sm]['std']:.0f}")
    if errs:
        print(f"  MAE: {np.mean(errs):.1f}%")

# ============================================================
# PLOT: 4×2 layout
# ============================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 24))
fig.suptitle('HAMi gpucores: Original vs AIMD×3 Patch\n'
             'k3s, gpu_burn 30s, RTX 4080 SUPER (2 runs each)',
             fontsize=14, fontweight='bold')

# --- Panel 1: Sweep accuracy with error bars ---
ax = axes[0, 0]
ax.set_title("Throughput vs Target (final proc'd / baseline)")
ax.plot(targets, targets, 'k--', alpha=0.5, linewidth=2, label='Ideal')

for name in runs:
    xs, ys, yerr = [], [], []
    for sm in targets:
        pct_mean = stats[name][sm]["mean"] / baseline * 100
        pct_std = stats[name][sm]["std"] / baseline * 100
        xs.append(sm)
        ys.append(pct_mean)
        yerr.append(pct_std)
    ax.errorbar(xs, ys, yerr=yerr, color=variant_colors[name], marker='o',
                linewidth=2, markersize=8, capsize=5, label=name)

ax.set_xlabel("gpucores Target (%)")
ax.set_ylabel("Actual Throughput (% of baseline)")
ax.legend(loc='upper left')
ax.set_xlim(10, 90)
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3)

# --- Panel 2: Error bars ---
ax = axes[0, 1]
ax.set_title("Error (Actual − Target)")
ax.axhline(0, color='k', linewidth=1, alpha=0.5)
width = 6
for i, name in enumerate(runs):
    xs, errs, yerr = [], [], []
    for sm in targets:
        pct_mean = stats[name][sm]["mean"] / baseline * 100
        pct_std = stats[name][sm]["std"] / baseline * 100
        xs.append(sm + (i - 0.5) * width)
        errs.append(pct_mean - sm)
        yerr.append(pct_std)
    ax.bar(xs, errs, width=width * 0.9, yerr=yerr, color=variant_colors[name],
           alpha=0.7, capsize=4, label=name)
ax.set_xlabel("gpucores Target (%)")
ax.set_ylabel("Error (percentage points)")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# --- Panel 3 & 4: Cumulative throughput (all levels) ---
for col, name in enumerate(runs):
    ax = axes[1, col]
    run_idx = 1 if len(runs[name]) > 1 else 0

    # Baseline (sm=0) curve
    base_ts = runs[name][run_idx][0][0]
    base_final = runs[name][run_idx][0][1]
    if len(base_ts) >= 2:
        barr = np.array(base_ts)
        ax.plot(barr[:, 0], barr[:, 1], color='#999999', linewidth=2,
                label=f'sm=0 ({base_final})')

    # Each gpucores level
    for sm in targets:
        test_ts = runs[name][run_idx][sm][0]
        test_final = runs[name][run_idx][sm][1]
        actual_pct = test_final / baseline * 100

        # Ideal line: baseline × target%
        if len(base_ts) >= 2:
            ax.plot(barr[:, 0], barr[:, 1] * sm / 100,
                    color=level_colors[sm], linestyle='--', linewidth=1, alpha=0.5)

        # Actual curve
        if len(test_ts) >= 2:
            tarr = np.array(test_ts)
            ax.plot(tarr[:, 0], tarr[:, 1], color=level_colors[sm], linewidth=2,
                    label=f'sm={sm}: {actual_pct:.0f}%')

    ax.set_title(f"Cumulative Throughput — {name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("proc'd (cumulative)")
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

# --- Panel 5 & 6: nvidia-smi SM Utilization (all levels) ---
for col, name in enumerate(runs):
    ax = axes[2, col]
    smi_for_variant = smi_runs.get(name, {})

    for sm in targets:
        smi = smi_for_variant.get(sm, [])
        if not smi:
            continue
        smi_arr = np.array(smi)
        mask30 = smi_arr[:, 0] <= 30.0
        times, utils = smi_arr[mask30, 0], smi_arr[mask30, 1]

        # Raw data (light)
        ax.plot(times, utils, color=level_colors[sm], alpha=0.3, linewidth=0.8)

        # Running avg 2s
        rt, rv = running_avg(times, utils)
        if rt:
            avg_val = np.mean(utils[int(len(utils)*0.2):])
            ax.plot(rt, rv, color=level_colors[sm], linewidth=2,
                    label=f'sm={sm} (steady: {avg_val:.0f}%)')

        # Target line
        ax.axhline(sm, color=level_colors[sm], linestyle='--', linewidth=1, alpha=0.4)

    ax.set_title(f"nvidia-smi SM Util — {name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SM Util (%)")
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

# --- Panel 7: MAE comparison ---
ax = axes[3, 0]
ax.set_title("Mean Absolute Error (MAE)")
mae_data = []
for name in runs:
    errs = []
    for sm in targets:
        pct = stats[name][sm]["mean"] / baseline * 100
        errs.append(abs(pct - sm))
    mae_data.append((name, np.mean(errs), variant_colors[name]))

names, maes, cols = zip(*mae_data)
bars = ax.barh(range(len(names)), maes, color=cols, alpha=0.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=12)
ax.set_xlabel("MAE (percentage points)")
for bar, mae in zip(bars, maes):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f'{mae:.1f}%', va='center', fontsize=14, fontweight='bold')
ax.set_xlim(0, max(maes) * 1.4)
ax.grid(True, alpha=0.3, axis='x')

# --- Panel 8: Data table ---
ax = axes[3, 1]
ax.set_title("Raw Data (2 runs each)")
ax.axis('off')

headers = ["Variant"] + [f"sm={sm}" for sm in levels] + ["MAE"]
table_data = []
for name in runs:
    row = [name]
    errs = []
    for sm in levels:
        mean = stats[name][sm]["mean"]
        std = stats[name][sm]["std"]
        pct = mean / baseline * 100
        if sm == 0:
            row.append(f"{mean:.0f}±{std:.0f}")
        else:
            row.append(f"{pct:.1f}±{std/baseline*100:.1f}%")
            errs.append(abs(pct - sm))
    row.append(f"{np.mean(errs):.1f}%")
    table_data.append(row)

table = ax.table(cellText=table_data, colLabels=headers,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.0)
for i, name in enumerate(runs):
    table[i + 1, 0].set_text_props(color=variant_colors[name], fontweight='bold')

plt.tight_layout()
outpath = "/tmp/gpu-bench-ts/k3s_final_comparison.png"
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {outpath}")
