#!/usr/bin/env python3
"""
Generate aggregated summary CSVs and publication-grade comparative plots
for CSP502 Lab Exercise 9.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "report" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.2,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save_fig(fig, stem: str):
    fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGS / f"{stem}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def summarize_and_plot_batch():
    df = pd.read_csv(DATA / "batch" / "batch_raw.csv")
    med = df.groupby(["dataset", "policy", "workers"], as_index=False).agg(
        median_sec=("elapsed_sec", "median"),
        median_thru=("throughput_rps", "median"),
        peak_heap_mb=("peak_heap_mb", "max"),
    )
    seq_map = med[med["policy"] == "sequential"].set_index("dataset")["median_sec"].to_dict()
    med["speedup"] = med.apply(lambda r: seq_map[r["dataset"]] / r["median_sec"], axis=1)
    med["efficiency"] = med["speedup"] / med["workers"]
    med.to_csv(DATA / "batch" / "batch_summary.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.7))
    for ds, lbl, col, mk in [
        ("medium_10k", "10k (FIFO)", "#C0392B", "o"),
        ("large_100k", "100k (FIFO)", "#1F618D", "s"),
        ("large_variant_100k", "100k Variant (FIFO)", "#27AE60", "^"),
    ]:
        sub = med[(med["dataset"] == ds) & (med["policy"] == "fifo")].sort_values("workers")
        ax1.plot(sub["workers"], sub["speedup"], marker=mk, lw=1.6, color=col, label=lbl)
        ax2.plot(sub["workers"], sub["efficiency"] * 100, marker=mk, lw=1.6, color=col, label=lbl)

    sub_p = med[(med["dataset"] == "large_100k") & (med["policy"] == "priority")].sort_values("workers")
    ax1.plot(sub_p["workers"], sub_p["speedup"], marker="D", lw=1.4, ls="--", color="#8E44AD", label="100k (Priority)")
    ax2.plot(sub_p["workers"], sub_p["efficiency"] * 100, marker="D", lw=1.4, ls="--", color="#8E44AD", label="100k (Priority)")

    ax1.set_title("Part C: Batch Speedup vs. Worker Count")
    ax1.set_xlabel("Worker Goroutines (W)")
    ax1.set_ylabel("Speedup (T_seq / T_conc)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([1, 2, 4, 8, 32])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

    ax2.set_title("Part C: Parallel Efficiency (W=32 Oversubscribed)")
    ax2.set_xlabel("Worker Goroutines (W)")
    ax2.set_ylabel("Efficiency (%)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([1, 2, 4, 8, 32])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "plot_partC_batch_scaling")


def plot_disturbances():
    df = pd.read_csv(DATA / "disturbances" / "disturbances_raw.csv")
    sc_order = ["D0_Normal", "D1_Burst", "D2_SlowTasks", "D3_TransientFail", "D4_PriorityPressure"]
    sc_lbls = ["D0\nNormal", "D1\nBurst", "D2\nSlow", "D3\nFail", "D4\nPrio 70%"]

    fifo = df[df["policy"] == "fifo"].set_index("scenario").loc[sc_order]
    prio = df[df["policy"] == "priority"].set_index("scenario").loc[sc_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7))
    x = np.arange(len(sc_order))
    w = 0.20

    ax1.bar(x - 1.5*w, fifo["high_p95_lat_ms"], width=w, color="#E74C3C", label="FIFO High p95")
    ax1.bar(x - 0.5*w, fifo["norm_p95_lat_ms"], width=w, color="#F39C12", label="FIFO Norm p95")
    ax1.bar(x + 0.5*w, prio["high_p95_lat_ms"], width=w, color="#1F618D", label="Priority High p95")
    ax1.bar(x + 1.5*w, prio["norm_p95_lat_ms"], width=w, color="#5DADE2", label="Priority Norm p95")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sc_lbls)
    ax1.set_ylabel("p95 Latency (ms)")
    ax1.set_title("Part D: Per-Class p95 Latency (W=4 Baseline)")
    ax1.set_ylim(0, max(fifo["norm_p95_lat_ms"].max(), prio["norm_p95_lat_ms"].max()) * 1.28)
    ax1.legend(loc="upper right", ncol=2, fontsize=7.6, frameon=True, framealpha=0.92)

    ax2.bar(x - 0.18, fifo["max_normal_wait_ms"], width=0.35, color="#C0392B", label="FIFO Max Normal Wait")
    ax2.bar(x + 0.18, prio["max_normal_wait_ms"], width=0.35, color="#1F618D", label="Priority Max Normal Wait")
    for i, (fv, pv) in enumerate(zip(fifo["max_normal_wait_ms"], prio["max_normal_wait_ms"])):
        ax2.text(i - 0.18, fv + 1.5, f"{fv:.1f}", ha="center", va="bottom", fontsize=7.2, color="#C0392B", fontweight="bold")
        ax2.text(i + 0.18, pv + 1.5, f"{pv:.1f}", ha="center", va="bottom", fontsize=7.2, color="#1F618D", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(sc_lbls)
    ax2.set_ylabel("Max Normal-Priority Wait (ms)")
    ax2.set_title("Part D: Normal-Class Starvation Indicator (W=4)")
    ax2.set_ylim(0, max(fifo["max_normal_wait_ms"].max(), prio["max_normal_wait_ms"].max()) * 1.32)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92)

    fig.tight_layout()
    save_fig(fig, "plot_partD_disturbances")


def summarize_and_plot_partf():
    df = pd.read_csv(DATA / "comparison" / "partf_raw.csv")
    sc_order = ["D0_Normal", "D1_Burst", "D2_SlowTasks", "D3_TransientFail", "D4_PriorityPressure"]
    pol_order = ["fifo", "priority", "fairness_only", "resilience_only", "modified"]
    pol_labels = {
        "fifo": "Orig-FIFO",
        "priority": "Orig-Priority",
        "fairness_only": "Fairness-Only",
        "resilience_only": "Resilience-Only",
        "modified": "Modified-Full",
    }

    summary = df.groupby(["scenario", "policy"], as_index=False).agg(
        successful_mean=("successful", "mean"),
        rejected_mean=("rejected", "mean"),
        timed_out_mean=("timed_out", "mean"),
        failed_mean=("failed", "mean"),
        retries_mean=("retries_used", "mean"),
        throughput_mean=("throughput_rps", "mean"),
        peak_q_mean=("peak_queue_len", "mean"),
        high_med_mean=("high_median_lat_ms", "mean"),
        high_p95_mean=("high_p95_lat_ms", "mean"),
        norm_med_mean=("norm_median_lat_ms", "mean"),
        norm_p95_mean=("norm_p95_lat_ms", "mean"),
        max_norm_wait_mean=("max_normal_wait_ms", "mean"),
        high_succ_rate_mean=("high_success_rate", "mean"),
        norm_succ_rate_mean=("norm_success_rate", "mean"),
        burst_rec_mean=("burst_recovery_ms", "mean"),
    )
    summary.to_csv(DATA / "comparison" / "partf_summary.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.9))
    x = np.arange(len(sc_order))
    sc_lbls = ["D0\nNormal", "D1\nBurst", "D2\nSlow", "D3\nFail", "D4\nPrio 70%"]
    colors = {
        "fifo": "#7F8C8D",
        "priority": "#C0392B",
        "fairness_only": "#E67E22",
        "resilience_only": "#8E44AD",
        "modified": "#1F618D",
    }
    w = 0.16

    for idx, pol in enumerate(pol_order):
        sub = summary[summary["policy"] == pol].set_index("scenario").loc[sc_order]
        offset = (idx - 2) * w
        ax1.bar(x + offset, sub["max_norm_wait_mean"], width=w, color=colors[pol], label=pol_labels[pol])
        ax2.bar(x + offset, sub["throughput_mean"], width=w, color=colors[pol], label=pol_labels[pol])

    ax1.axhline(60.0, color="black", ls=":", lw=1.1, label="SLA Limit (60ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sc_lbls)
    ax1.set_ylabel("Max Normal-Priority Wait (ms)")
    ax1.set_title("Part F: Normal Wait & Starvation (W=2, K=64)")
    ax1.legend(loc="upper left", ncol=2, fontsize=7.2, frameon=True, framealpha=0.92)

    ax2.set_xticks(x)
    ax2.set_xticklabels(sc_lbls)
    ax2.set_ylabel("Successful Throughput (req/s)")
    ax2.set_title("Part F: Successful Throughput (W=2, K=64)")
    ax2.legend(loc="lower left", ncol=2, fontsize=7.2, frameon=True, framealpha=0.92)

    fig.tight_layout()
    save_fig(fig, "plot_partF_comparison")


def main():
    summarize_and_plot_batch()
    plot_disturbances()
    summarize_and_plot_partf()
    print("Generated batch_summary.csv, partf_summary.csv, and 3 comparative figures in report/figures/")


if __name__ == "__main__":
    main()
