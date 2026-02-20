#!/usr/bin/env python3
"""Final comparison: Original vs Orig + AIMD×3 (2 runs each)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import csv

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

def calc_instantaneous_util(ts_data, baseline_final, window=1.0):
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
                rt.append(t)
                rv.append((dc / dt) / baseline_rate * 100)
    return rt, rv

def get_stall_intervals(ts_data, min_stall_count=3):
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

# ============================================================
# Load data: 2 runs each
# ============================================================
basedir = "/tmp/gpu-bench-ts/k3s"
levels = [0, 20, 40, 60, 80]

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

smi_data = {
    "Original (v2.8.0)": [
        {sm: parse_smi_csv(f"{basedir}/stock2_sm{sm}_smi.csv") for sm in levels},
    ],
    "Orig + AIMD×3": [
        {sm: parse_smi_csv(f"{basedir}/origv5b_sm{sm}_smi.csv") for sm in levels},
    ],
}

colors = {"Original (v2.8.0)": "#d62728", "Orig + AIMD×3": "#1f77b4"}

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
# PLOT: 3×2 layout
# ============================================================
fig, axes = plt.subplots(4, 2, figsize=(16, 24))
fig.suptitle('HAMi gpucores: Original vs AIMD×3 Patch\n'
             'k3s, gpu_burn 30s, RTX 4080 SUPER (2 runs each)',
             fontsize=14, fontweight='bold')

# --- Panel 1: Sweep accuracy with error bars ---
ax = axes[0, 0]
ax.set_title("Throughput vs Target")
targets = [sm for sm in levels if sm > 0]
ax.plot(targets, targets, 'k--', alpha=0.5, linewidth=2, label='Ideal')

for name in runs:
    xs, ys, yerr = [], [], []
    for sm in targets:
        pct_mean = stats[name][sm]["mean"] / baseline * 100
        pct_std = stats[name][sm]["std"] / baseline * 100
        xs.append(sm)
        ys.append(pct_mean)
        yerr.append(pct_std)
    ax.errorbar(xs, ys, yerr=yerr, color=colors[name], marker='o',
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
    ax.bar(xs, errs, width=width * 0.9, yerr=yerr, color=colors[name],
           alpha=0.7, capsize=4, label=name)
ax.set_xlabel("gpucores Target (%)")
ax.set_ylabel("Error (percentage points)")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# --- Panel 3: Time-series at sm=40 (instantaneous throughput) ---
sm = 40
ax = axes[1, 0]
ax.set_title(f"Instantaneous Throughput at gpucores={sm} (1s window)")
# Use latest run for time-series
for name, smi_list in smi_data.items():
    run_idx = 1 if len(runs[name]) > 1 else 0
    ts_data = runs[name][run_idx][sm][0]
    base_final = runs[name][run_idx][0][1]
    rt, rv = calc_instantaneous_util(ts_data, base_final)
    if rt:
        ax.plot(rt, rv, color=colors[name], alpha=0.8, linewidth=1, label=name)
        stalls = get_stall_intervals(ts_data)
        for j, (t0, t1) in enumerate(stalls):
            ax.axvspan(t0, t1, alpha=0.08, color=colors[name],
                       label='OFF interval' if j == 0 and name == "Orig + AIMD×3" else None)
ax.axhline(sm, color='k', linestyle='--', alpha=0.5, label=f'Target {sm}%')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Throughput (% of baseline)")
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 120)
ax.grid(True, alpha=0.3)

# --- Panel 4: Cumulative SM utilization ---
ax = axes[1, 1]
ax.set_title(f"Cumulative SM Utilization at gpucores={sm} (running avg)")
for name, smi_list in smi_data.items():
    smi = smi_list[0][sm]
    if smi:
        smi_arr = np.array(smi)
        times = smi_arr[:, 0]
        utils = smi_arr[:, 1]
        cum_avg = np.cumsum(utils) / np.arange(1, len(utils) + 1)
        final_avg = cum_avg[-1]
        ax.plot(times, cum_avg, color=colors[name], linewidth=2,
                label=f'{name} ({final_avg:.1f}%)')
ax.axhline(sm, color='k', linestyle='--', alpha=0.5, label=f'Target {sm}%')
ax.set_xlabel("Time (s)")
ax.set_ylabel("SM Utilization (% running avg)")
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3)

# --- Panel 5: Raw nvidia-smi SM utilization ---
ax = axes[2, 0]
ax.set_title(f"nvidia-smi SM Utilization at gpucores={sm}")
for name, smi_list in smi_data.items():
    smi = smi_list[0][sm]
    if smi:
        smi_arr = np.array(smi)
        ax.plot(smi_arr[:, 0], smi_arr[:, 1], color=colors[name], alpha=0.6,
                linewidth=0.8, label=name)
ax.axhline(sm, color='k', linestyle='--', alpha=0.5, label=f'Target {sm}%')
ax.set_xlabel("Time (s)")
ax.set_ylabel("SM Utilization (%)")
ax.legend(loc='upper right', fontsize=9)
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3)

# --- Panel 6: Cumulative throughput ---
ax = axes[2, 1]
ax.set_title(f"Cumulative Throughput at gpucores={sm}")
for name, smi_list in smi_data.items():
    run_idx = 1 if len(runs[name]) > 1 else 0
    d = runs[name][run_idx][sm]
    if d[0]:
        arr = np.array(d[0])
        cum_pct = arr[:, 1] / baseline * 100
        final_pct = d[1] / baseline * 100
        ax.plot(arr[:, 0], cum_pct, color=colors[name], linewidth=2,
                label=f'{name} ({final_pct:.1f}%)')
ax.axhline(sm, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Cumulative (% of baseline)")
ax.legend(loc='upper left', fontsize=9)
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
    mae_data.append((name, np.mean(errs), colors[name]))

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
    table[i + 1, 0].set_text_props(color=colors[name], fontweight='bold')

plt.tight_layout()
outpath = "/tmp/gpu-bench-ts/k3s_final_comparison.png"
plt.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {outpath}")
