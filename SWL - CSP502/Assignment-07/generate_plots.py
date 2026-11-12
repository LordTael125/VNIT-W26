import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set matplotlib publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

df = pd.read_csv('benchmark_results.csv')
recovery_df = pd.read_csv('recovery_timeseries.csv')

# -------------------------------------------------------------------------------------------------
# Plot 1: Latency Profile Under Adverse Conditions (Baseline vs Improved)
# -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))

scenarios = ['Baseline_Normal', 'Improved_Normal', 
             'Baseline_C2_SlowTasks', 'Improved_C2_SlowTasks',
             'Baseline_CombinedStress', 'Improved_CombinedStress']

labels = ['Normal\n(Base)', 'Normal\n(Impr)', 
          'Slow Tasks\n(Base)', 'Slow Tasks\n(Impr)',
          'Combined\n(Base)', 'Combined\n(Impr)']

sub_df = df.set_index('Scenario').loc[scenarios].reset_index()

x = np.arange(len(scenarios))
width = 0.25

rects1 = ax.bar(x - width, sub_df['MeanResponseTime_ms'], width, label='Mean Latency', color='#2b5c8f')
rects2 = ax.bar(x, sub_df['P99ResponseTime_ms'], width, label='P99 Latency', color='#d95f02')
rects3 = ax.bar(x + width, sub_df['MaxResponseTime_ms'], width, label='Max Latency', color='#7570b3')

ax.set_ylabel('Response Time (ms) [Log Scale]')
ax.set_yscale('log')
ax.set_title('Response Time Distribution: Baseline vs. Resilient Architecture')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(frameon=True, loc='upper left')
ax.grid(True, which='both', axis='y')

# Annotate values on top of Combined Stress bars
ax.annotate(f"{sub_df.loc[4, 'MeanResponseTime_ms']:.1f}ms", 
            xy=(4 - width, sub_df.loc[4, 'MeanResponseTime_ms']), 
            xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7, fontweight='bold')
ax.annotate(f"{sub_df.loc[5, 'MeanResponseTime_ms']:.1f}ms", 
            xy=(5 - width, sub_df.loc[5, 'MeanResponseTime_ms']), 
            xytext=(0, 3), textcoords="offset points", ha='center', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_latency_comparison.pdf')
plt.savefig('plot_latency_comparison.png')
plt.close()
print("[+] Generated plot_latency_comparison.pdf & png")

# -------------------------------------------------------------------------------------------------
# Plot 2: Transient Queue Dynamics & Recovery Profile
# -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2))

base_rec = recovery_df[recovery_df['Architecture'] == 'Baseline']
impr_rec = recovery_df[recovery_df['Architecture'] == 'Improved']

ax.plot(base_rec['TimestampSec'], base_rec['QueueDepth'], 
        label='Baseline (Unbounded Queue)', color='#d95f02', linewidth=2)
ax.plot(impr_rec['TimestampSec'], impr_rec['QueueDepth'], 
        label='Improved (Bounded Cap=40 + Timeout)', color='#1b9e77', linewidth=2, linestyle='--')

ax.axvspan(1.0, 2.0, color='gray', alpha=0.15, label='Surge Disturbance Window')
ax.axhline(40, color='red', linestyle=':', alpha=0.7, label='Admission Limit (K=40)')

ax.set_xlabel('Elapsed Time (seconds)')
ax.set_ylabel('Queue Depth (pending requests)')
ax.set_title('Queue Depth Dynamics: Surge Absorption and Post-Disturbance Recovery')
ax.legend(frameon=True, loc='upper right')
ax.grid(True)

plt.tight_layout()
plt.savefig('plot_queue_recovery.pdf')
plt.savefig('plot_queue_recovery.png')
plt.close()
print("[+] Generated plot_queue_recovery.pdf & png")

# -------------------------------------------------------------------------------------------------
# Plot 3: Request Outcome Breakdown under Stress
# -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))

comp_scenarios = ['Baseline_Normal', 'Improved_Normal',
                  'Baseline_C2_SlowTasks', 'Improved_C2_SlowTasks',
                  'Baseline_CombinedStress', 'Improved_CombinedStress']

comp_labels = ['Normal\n(Base)', 'Normal\n(Impr)', 
               'Slow Tasks\n(Base)', 'Slow Tasks\n(Impr)',
               'Combined\n(Base)', 'Combined\n(Impr)']

c_df = df.set_index('Scenario').loc[comp_scenarios].reset_index()

x = np.arange(len(comp_scenarios))
width = 0.55

p1 = ax.bar(x, c_df['Successful'], width, label='Successful', color='#2ca02c')
p2 = ax.bar(x, c_df['TimedOut'], width, bottom=c_df['Successful'], label='TimedOut (SLA Miss)', color='#ff7f0e')
p3 = ax.bar(x, c_df['Failed'], width, bottom=c_df['Successful'] + c_df['TimedOut'], label='Failed (Fault)', color='#d62728')
p4 = ax.bar(x, c_df['Rejected'], width, bottom=c_df['Successful'] + c_df['TimedOut'] + c_df['Failed'], label='Rejected (Load Shed)', color='#9467bd')

ax.set_ylabel('Request Count')
ax.set_title('Request Outcome Breakdown: Controlled Shedding vs. Uncontrolled Starvation')
ax.set_xticks(x)
ax.set_xticklabels(comp_labels)
ax.legend(frameon=True, loc='lower left')
ax.grid(True, axis='y')

# Annotate totals
for i in x:
    tot = c_df.loc[i, 'TotalGenerated']
    ax.text(i, tot + 15, f"{tot}", ha='center', fontsize=8)

ax.set_ylim(0, 1400)
plt.tight_layout()
plt.savefig('plot_outcomes_breakdown.pdf')
plt.savefig('plot_outcomes_breakdown.png')
plt.close()
print("[+] Generated plot_outcomes_breakdown.pdf & png")

# -------------------------------------------------------------------------------------------------
# Plot 4: Throughput & Goodput Comparison
# -------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))

t_df = df.set_index('Scenario').loc[['Baseline_Normal', 'Improved_Normal',
                                     'Baseline_C1_Surge', 'Improved_C1_Surge',
                                     'Baseline_CombinedStress', 'Improved_CombinedStress']].reset_index()

t_labels = ['Normal\n(Base)', 'Normal\n(Impr)',
            'Surge\n(Base)', 'Surge\n(Impr)',
            'Stress\n(Base)', 'Stress\n(Impr)']

x = np.arange(len(t_labels))
width = 0.35

r1 = ax.bar(x - width/2, t_df['Throughput_req_s'], width, label='Total Throughput (req/s)', color='#1f77b4')
r2 = ax.bar(x + width/2, t_df['Goodput_req_s'], width, label='Effective Goodput (req/s)', color='#2ca02c')

ax.set_ylabel('Rate (requests / second)')
ax.set_title('Service Throughput vs. Goodput Under Normal and Disturbed Regimes')
ax.set_xticks(x)
ax.set_xticklabels(t_labels)
ax.legend(frameon=True, loc='upper left')
ax.grid(True, axis='y')

plt.tight_layout()
plt.savefig('plot_throughput_goodput.pdf')
plt.savefig('plot_throughput_goodput.png')
plt.close()
print("[+] Generated plot_throughput_goodput.pdf & png")

