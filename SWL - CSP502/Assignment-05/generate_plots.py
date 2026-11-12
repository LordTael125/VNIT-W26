import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

df = pd.read_csv('benchmark_results.csv')

# -------------------------------------------------------------------------
# Figure 1: Full-scale Growth on Random Data (Log-Log Scale)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.4))
sub_rand = df[df['Distribution'] == 'Random']
ins = sub_rand[sub_rand['Algorithm'] == 'InsertionSort']
mrg = sub_rand[sub_rand['Algorithm'] == 'MergeSort']

ax.plot(ins['InputSize'], ins['Median_us'], 'o-', color='#d9534f', linewidth=2, markersize=5, label=r'Insertion Sort ($O(n^2)$)')
ax.plot(mrg['InputSize'], mrg['Median_us'], 's--', color='#0275d8', linewidth=2, markersize=5, label=r'Merge Sort ($O(n \log n)$)')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Input Size ($n$) [Log Scale]')
ax.set_ylabel(r'Execution Time ($\mu$s) [Log Scale]')
ax.set_title('Execution Time vs. Input Size (Random Integers)')
ax.grid(True, which="both", ls="--", alpha=0.5)

# Annotate crossover
ax.axvspan(100, 200, color='#f0ad4e', alpha=0.25, label=r'Crossover Region ($n \approx 140$)')
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('plot_growth_random.pdf')
plt.savefig('plot_growth_random.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 2: Zoom on Small-Input Crossover Region (Linear Scale, n <= 1000)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.4))
ins_zoom = ins[ins['InputSize'] <= 1000]
mrg_zoom = mrg[mrg['InputSize'] <= 1000]

ax.plot(ins_zoom['InputSize'], ins_zoom['Median_us'], 'o-', color='#d9534f', linewidth=2, markersize=5, label='Insertion Sort')
ax.plot(mrg_zoom['InputSize'], mrg_zoom['Median_us'], 's--', color='#0275d8', linewidth=2, markersize=5, label='Merge Sort')

ax.set_xlabel(r'Input Size ($n$)')
ax.set_ylabel(r'Execution Time ($\mu$s)')
ax.set_title(r'Small-Input Regime & Crossover Investigation ($n \leq 1000$)')
ax.grid(True, ls="--", alpha=0.5)

# Highlight crossover point around n=140
ax.annotate(r'Crossover ($n \approx 140$)', xy=(140, 3.2), xytext=(250, 0.5),
            arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5),
            fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#ffeeba"))

ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('plot_crossover_zoom.pdf')
plt.savefig('plot_crossover_zoom.png', dpi=300)
plt.close()

# -------------------------------------------------------------------------
# Figure 3: Input Distribution Impact (Sorted vs Random vs Reverse)
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

# Left: Insertion Sort across distributions
for dist, col, m in [('Sorted', '#5cb85c', '^'), ('Nearly Sorted', '#f0ad4e', 'd'), ('Random', '#0275d8', 'o'), ('Reverse', '#d9534f', 's')]:
    sub = df[(df['Distribution'] == dist) & (df['Algorithm'] == 'InsertionSort')]
    ax1.plot(sub['InputSize'], sub['Median_us'], marker=m, color=col, label=dist, linewidth=1.5, markersize=4)

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel(r'Input Size ($n$)')
ax1.set_ylabel(r'Execution Time ($\mu$s)')
ax1.set_title('Insertion Sort: Data Sensitivity')
ax1.grid(True, which="both", ls="--", alpha=0.4)
ax1.legend(fontsize=8, frameon=True)

# Right: Merge Sort across distributions
for dist, col, m in [('Sorted', '#5cb85c', '^'), ('Nearly Sorted', '#f0ad4e', 'd'), ('Random', '#0275d8', 'o'), ('Reverse', '#d9534f', 's')]:
    sub = df[(df['Distribution'] == dist) & (df['Algorithm'] == 'MergeSort')]
    ax2.plot(sub['InputSize'], sub['Median_us'], marker=m, color=col, label=dist, linewidth=1.5, markersize=4)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Input Size ($n$)')
ax2.set_ylabel(r'Execution Time ($\mu$s)')
ax2.set_title('Merge Sort: Data Sensitivity')
ax2.grid(True, which="both", ls="--", alpha=0.4)
ax2.legend(fontsize=8, frameon=True)

plt.tight_layout()
plt.savefig('plot_distributions.pdf')
plt.savefig('plot_distributions.png', dpi=300)
plt.close()

print("All plots generated successfully: plot_growth_random.pdf, plot_crossover_zoom.pdf, plot_distributions.pdf")
