import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Apply professional publication styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

df = pd.read_csv('benchmark_results.csv')

# Palette
colors = {
    10000: '#7f7f7f',
    50000: '#17becf',
    200000: '#2ca02c',
    1000000: '#ff7f0e',
    5000000: '#9467bd',
    10000000: '#1f77b4'
}

# -------------------------------------------------------------------------
# Figure 1: Speedup vs. Worker Count (Sorting vs Statistical Analysis)
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)

workers = [1, 2, 4, 8, 12, 16, 24]

# Left: Sorting
df_sort = df[df['Operation'] == 'Sorting']
for sz in [10000, 200000, 1000000, 10000000]:
    sub = df_sort[df_sort['InputSize'] == sz].sort_values('Workers')
    label = f'N = {sz:,}'
    ax1.plot(sub['Workers'], sub['Speedup'], marker='o', linewidth=1.8, markersize=5,
             color=colors[sz], label=label)

# Ideal linear speedup line
ax1.plot([1, 8], [1, 8], 'k:', alpha=0.4, label='Ideal Linear (W)')
ax1.axvline(16, color='red', linestyle='--', alpha=0.4, label='Logical CPUs (16)')
ax1.set_xlabel('Worker Goroutines ($W$)')
ax1.set_ylabel('Speedup $S(W) = T_{seq} / T_{par}$')
ax1.set_title('Operation A: Merge Sort Speedup')
ax1.set_xticks(workers)
ax1.set_ylim(0, 8.5)
ax1.grid(True, ls="--", alpha=0.5)
ax1.legend(loc='upper left', frameon=True, fontsize=8)

# Right: Statistical Analysis
df_stats = df[df['Operation'] == 'Statistical Analysis']
for sz in [10000, 200000, 1000000, 10000000]:
    sub = df_stats[df_stats['InputSize'] == sz].sort_values('Workers')
    label = f'N = {sz:,}'
    ax2.plot(sub['Workers'], sub['Speedup'], marker='s', linewidth=1.8, markersize=5,
             color=colors[sz], label=label)

ax2.plot([1, 8], [1, 8], 'k:', alpha=0.4, label='Ideal Linear (W)')
ax2.axvline(16, color='red', linestyle='--', alpha=0.4, label='Logical CPUs (16)')
ax2.set_xlabel('Worker Goroutines ($W$)')
ax2.set_title('Operation B: Statistical Analysis Speedup')
ax2.set_xticks(workers)
ax2.grid(True, ls="--", alpha=0.5)
ax2.legend(loc='upper left', frameon=True, fontsize=8)

plt.tight_layout()
plt.savefig('plot_speedup_workers.pdf')
plt.savefig('plot_speedup_workers.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 2: Parallel Efficiency Decay vs. Worker Count
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)

for sz in [10000, 200000, 1000000, 10000000]:
    sub = df_sort[df_sort['InputSize'] == sz].sort_values('Workers')
    ax1.plot(sub['Workers'], sub['Efficiency'], marker='o', linewidth=1.8, markersize=5,
             color=colors[sz], label=f'N = {sz:,}')

ax1.axhline(1.0, color='black', linestyle=':', alpha=0.5, label='100% Efficiency')
ax1.axvline(16, color='red', linestyle='--', alpha=0.4, label='Logical CPUs (16)')
ax1.set_xlabel('Worker Goroutines ($W$)')
ax1.set_ylabel(r'Parallel Efficiency $E(W) = S(W) / W$')
ax1.set_title('Operation A: Sorting Efficiency Decay')
ax1.set_xticks(workers)
ax1.set_ylim(0, 1.08)
ax1.grid(True, ls="--", alpha=0.5)
ax1.legend(loc='upper right', frameon=True, fontsize=8)

for sz in [10000, 200000, 1000000, 10000000]:
    sub = df_stats[df_stats['InputSize'] == sz].sort_values('Workers')
    ax2.plot(sub['Workers'], sub['Efficiency'], marker='s', linewidth=1.8, markersize=5,
             color=colors[sz], label=f'N = {sz:,}')

ax2.axhline(1.0, color='black', linestyle=':', alpha=0.5, label='100% Efficiency')
ax2.axvline(16, color='red', linestyle='--', alpha=0.4, label='Logical CPUs (16)')
ax2.set_xlabel('Worker Goroutines ($W$)')
ax2.set_title('Operation B: Stats Efficiency Decay')
ax2.set_xticks(workers)
ax2.grid(True, ls="--", alpha=0.5)
ax2.legend(loc='upper right', frameon=True, fontsize=8)

plt.tight_layout()
plt.savefig('plot_efficiency_decay.pdf')
plt.savefig('plot_efficiency_decay.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 3: Execution Time vs Input Size (Log-Log Scale)
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4))

worker_styles = {
    1: ('#000000', '-', 'o', 'Sequential (W=1)'),
    2: ('#2ca02c', '--', '^', 'Parallel W=2'),
    4: ('#ff7f0e', '-.', 's', 'Parallel W=4'),
    8: ('#1f77b4', ':', 'd', 'Parallel W=8'),
    16: ('#d62728', '-', 'v', 'Parallel W=16')
}

# Left: Sorting
for w, (c, ls, m, lbl) in worker_styles.items():
    sub = df_sort[df_sort['Workers'] == w].sort_values('InputSize')
    ax1.plot(sub['InputSize'], sub['MeanTimeMs'], linestyle=ls, marker=m, color=c,
             linewidth=1.6, markersize=4.5, label=lbl)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Input Size ($N$) [Log Scale]')
ax1.set_ylabel('Execution Time (ms) [Log Scale]')
ax1.set_title('Sorting: Time vs Input Size')
ax1.grid(True, which="both", ls="--", alpha=0.4)
ax1.legend(loc='upper left', frameon=True, fontsize=8)

# Right: Stats
for w, (c, ls, m, lbl) in worker_styles.items():
    sub = df_stats[df_stats['Workers'] == w].sort_values('InputSize')
    ax2.plot(sub['InputSize'], sub['MeanTimeMs'], linestyle=ls, marker=m, color=c,
             linewidth=1.6, markersize=4.5, label=lbl)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Input Size ($N$) [Log Scale]')
ax2.set_ylabel('Execution Time (ms) [Log Scale]')
ax2.set_title('Statistical Analysis: Time vs Input Size')
ax2.grid(True, which="both", ls="--", alpha=0.4)
ax2.legend(loc='upper left', frameon=True, fontsize=8)

plt.tight_layout()
plt.savefig('plot_execution_time_scaling.pdf')
plt.savefig('plot_execution_time_scaling.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 4: Amdahl's Law Theoretical Model vs Empirical Speedup (N = 10,000,000)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 3.4))

w_range = np.linspace(1, 24, 100)

def amdahl(p_frac, w):
    return 1.0 / ((1.0 - p_frac) + (p_frac / w))

# Theoretical curves
for p_frac, ls, c in [(0.75, ':', '#7f7f7f'), (0.80, '--', '#bcbd22'), (0.88, '-.', '#ff7f0e'), (0.95, '-', '#17becf')]:
    ax.plot(w_range, [amdahl(p_frac, w) for w in w_range], linestyle=ls, color=c,
            alpha=0.85, label=f"Amdahl P = {int(p_frac*100)}% (Max: {1/(1-p_frac):.1f}x)")

# Empirical points for N = 10,000,000
sub_s = df_sort[df_sort['InputSize'] == 10000000].sort_values('Workers')
sub_a = df_stats[df_stats['InputSize'] == 10000000].sort_values('Workers')

ax.plot(sub_s['Workers'], sub_s['Speedup'], 'o-', color='#d62728', linewidth=2.2, markersize=6,
        label=r'Empirical Sort ($N=10^7$)')
ax.plot(sub_a['Workers'], sub_a['Speedup'], 's-', color='#1f77b4', linewidth=2.2, markersize=6,
        label=r'Empirical Stats ($N=10^7$)')

ax.axvline(16, color='black', linestyle='--', alpha=0.35, label='Physical/Logical Limit (16 CPUs)')
ax.set_xlabel('Processors / Worker Goroutines ($W$)')
ax.set_ylabel('Speedup $S(W)$')
ax.set_title('Amdahl\'s Law Model vs. Empirical Measurements ($N = 10^7$)')
ax.set_xticks(workers)
ax.set_ylim(0.5, 9.5)
ax.grid(True, ls="--", alpha=0.5)
ax.legend(loc='upper left', frameon=True, fontsize=8)

plt.tight_layout()
plt.savefig('plot_amdahl_fit.pdf')
plt.savefig('plot_amdahl_fit.png', dpi=300)
plt.close()

print("All 4 analytical plots successfully generated in PDF and PNG formats.")
