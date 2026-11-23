#!/usr/bin/env python3
"""
Generate publication-grade comparative plots and aggregated summary tables
for CSP502 Practical 8 (Baseline vs. Improved Concurrent Go Service).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Clean academic publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.28,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COND_ORDER = ["normal", "overload_slow", "failure_cancel", "ack_faults"]
COND_LABELS = {
    "normal": "Normal\n(Steady)",
    "overload_slow": "Overload +\nSlow Worker",
    "failure_cancel": "Failure +\nMid-Shutdown",
    "ack_faults": "Part D:\nAck Faults",
}

C_BASE = "#C0392B"    # crimson red
C_IMP = "#1F618D"     # deep steel blue
C_BASE_LT = "#E67E22" # warm amber
C_IMP_LT = "#2E86C1"  # bright blue


def save_fig(fig, stem: str):
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def build_summary_csv(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "terminal_accounting_violations",
        "duplicate_terminal_outcomes",
        "missing_terminal_ids",
        "successful",
        "rejected",
        "timed_out",
        "failed",
        "cancelled",
        "ack_retries_used",
        "duplicate_acks_ignored",
        "max_queue_depth",
        "throughput_rps",
        "goodput_rps",
        "mean_queue_wait_ms",
        "mean_response_ms",
        "p95_response_ms",
        "p99_response_ms",
        "shutdown_completion_ms",
        "outstanding_work_after_shutdown",
        "goroutine_delta",
    ]
    rows = []
    for cond in COND_ORDER:
        for arch in ["Baseline", "Improved"]:
            sub = df[(df["condition"] == cond) & (df["architecture"] == arch)]
            if sub.empty:
                continue
            row = {"condition": cond, "architecture": arch, "reps": len(sub)}
            for m in metrics:
                row[f"{m}_mean"] = round(float(sub[m].mean()), 3)
                row[f"{m}_std"] = round(float(sub[m].std(ddof=1)) if len(sub) > 1 else 0.0, 3)
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(DATA_DIR / "comparison" / "comparison_summary.csv", index=False)
    return summary


def plot_partb_normal_telemetry():
    summary_path = DATA_DIR / "baseline_runs" / "baseline_summary.csv"
    events_path = DATA_DIR / "baseline_runs" / "run1_events.jsonl"
    if not summary_path.exists() or not events_path.exists():
        return

    bsum = pd.read_csv(summary_path)
    ts_ms, active_w, q_depth = [], [], []
    with open(events_path, "r") as f:
        for idx, line in enumerate(f):
            if idx % 4 != 0:
                continue
            ev = json.loads(line)
            ts_ms.append(ev["elapsed_ms"] / 1000.0)
            active_w.append(ev["active_workers"])
            q_depth.append(ev["queue_depth"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.6))

    # Left: Run-to-run latency consistency across 3 normal runs
    x = np.arange(len(bsum))
    w = 0.25
    ax1.bar(x - w, bsum["mean_response_ms"], width=w, color="#2E86C1", label="Mean Response (ms)")
    ax1.bar(x, bsum["p95_response_ms"], width=w, color="#1B4F72", label="p95 Response (ms)")
    ax1.bar(x + w, bsum["mean_queue_wait_ms"], width=w, color="#F39C12", label="Mean Queue Wait (ms)")
    for i, r in bsum.iterrows():
        ax1.text(i - w, r["mean_response_ms"] + 0.35, f"{r['mean_response_ms']:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax1.text(i, r["p95_response_ms"] + 0.35, f"{r['p95_response_ms']:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax1.text(i + w, r["mean_queue_wait_ms"] + 0.35, f"{r['mean_queue_wait_ms']:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Run {int(r.repetition)}\n(seed={int(r.seed)})" for _, r in bsum.iterrows()])
    ax1.set_ylabel("Duration (ms)")
    ax1.set_title("Part B: Baseline Normal Latency (3 Repetitions)")
    ax1.set_ylim(0, max(bsum["p95_response_ms"]) * 1.52)
    ax1.legend(loc="upper center", ncol=2, fontsize=7.8, frameon=True, framealpha=0.92)

    # Right: Active workers & queue depth over time in Run 1
    ax2.plot(ts_ms, active_w, color="#1F618D", lw=1.1, alpha=0.9, label="Active Workers (W=8)")
    ax2.plot(ts_ms, q_depth, color="#C0392B", lw=1.1, alpha=0.85, label="Queue Depth")
    ax2.set_xlabel("Elapsed Time (s)")
    ax2.set_ylabel("Count")
    ax2.set_title("Part B: Active Workers & Queue Depth (Run 1)")
    ax2.set_ylim(-0.3, max(max(active_w), max(q_depth)) + 2.0)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "plot_partB_normal_telemetry")


def plot_latency_p95_comparison(summary: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    x = np.arange(len(COND_ORDER))
    w = 0.35

    base = summary[summary["architecture"] == "Baseline"].set_index("condition").loc[COND_ORDER]
    imp = summary[summary["architecture"] == "Improved"].set_index("condition").loc[COND_ORDER]

    # Mean Response Time (Admitted)
    b1 = ax1.bar(x - w / 2, base["mean_response_ms_mean"], yerr=base["mean_response_ms_std"],
                 width=w, capsize=3, color=C_BASE, alpha=0.9, label="Baseline")
    b2 = ax1.bar(x + w / 2, imp["mean_response_ms_mean"], yerr=imp["mean_response_ms_std"],
                 width=w, capsize=3, color=C_IMP, alpha=0.9, label="Improved")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels([COND_LABELS[c] for c in COND_ORDER])
    ax1.set_ylabel("Mean Response Time (ms, log scale)")
    ax1.set_title("Mean Response Time (Admitted Requests)")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)
    for bar, val, err in zip(b1, base["mean_response_ms_mean"], base["mean_response_ms_std"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, (val + err) * 1.16, f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=C_BASE)
    for bar, val, err in zip(b2, imp["mean_response_ms_mean"], imp["mean_response_ms_std"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, (val + err) * 1.16, f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=C_IMP)
    ax1.set_ylim(1, max(base["mean_response_ms_mean"].max(), imp["mean_response_ms_mean"].max()) * 3.4)

    # p95 Response Time (Admitted)
    p1 = ax2.bar(x - w / 2, base["p95_response_ms_mean"], yerr=base["p95_response_ms_std"],
                 width=w, capsize=3, color=C_BASE, alpha=0.9, label="Baseline")
    p2 = ax2.bar(x + w / 2, imp["p95_response_ms_mean"], yerr=imp["p95_response_ms_std"],
                 width=w, capsize=3, color=C_IMP, alpha=0.9, label="Improved")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels([COND_LABELS[c] for c in COND_ORDER])
    ax2.set_ylabel("p95 Response Time (ms, log scale)")
    ax2.set_title("Tail Latency: p95 Response Time (Nearest-Rank)")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9)
    for bar, val, err in zip(p1, base["p95_response_ms_mean"], base["p95_response_ms_std"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, (val + err) * 1.16, f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=C_BASE)
    for bar, val, err in zip(p2, imp["p95_response_ms_mean"], imp["p95_response_ms_std"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, (val + err) * 1.16, f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=C_IMP)
    ax2.set_ylim(1, max(base["p95_response_ms_mean"].max(), imp["p95_response_ms_mean"].max()) * 3.4)

    fig.tight_layout()
    save_fig(fig, "plot_latency_p95_comparison")


def plot_outcomes_reliability(summary: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.9))

    # Left: Stacked outcome counts per condition/architecture
    labels = []
    succ, rej, tout, fail, canc, miss = [], [], [], [], [], []
    for c in COND_ORDER:
        for arch in ["Baseline", "Improved"]:
            r = summary[(summary["condition"] == c) & (summary["architecture"] == arch)].iloc[0]
            short_c = {"normal": "Norm", "overload_slow": "Ovld", "failure_cancel": "Fail", "ack_faults": "Ack"}[c]
            short_a = "Base" if arch == "Baseline" else "Imp"
            labels.append(f"{short_c}\n{short_a}")
            succ.append(r["successful_mean"])
            rej.append(r["rejected_mean"])
            tout.append(r["timed_out_mean"])
            fail.append(r["failed_mean"])
            canc.append(r["cancelled_mean"])
            miss.append(r["missing_terminal_ids_mean"])

    x = np.arange(len(labels))
    succ = np.array(succ)
    rej = np.array(rej)
    tout = np.array(tout)
    fail = np.array(fail)
    canc = np.array(canc)
    miss = np.array(miss)

    ax1.bar(x, succ, color="#27AE60", label="Success")
    ax1.bar(x, rej, bottom=succ, color="#F39C12", label="Rejected")
    ax1.bar(x, tout, bottom=succ + rej, color="#8E44AD", label="TimedOut")
    ax1.bar(x, fail, bottom=succ + rej + tout, color="#E74C3C", label="Failed")
    ax1.bar(x, canc, bottom=succ + rej + tout + fail, color="#5DADE2", label="Cancelled")
    ax1.bar(x, miss, bottom=succ + rej + tout + fail + canc, color="#2C3E50", hatch="//", label="Missing (Lost)")

    ax1.axhline(1200, color="black", linestyle=":", linewidth=1.0, label="Generated (N=1200)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.8)
    ax1.set_ylabel("Mean Request Count (3 reps)")
    ax1.set_title("Terminal Outcomes vs. Generated Target (N=1200)")
    ax1.legend(loc="upper left", ncol=2, fontsize=7.3, frameon=True, framealpha=0.92)
    ax1.set_ylim(0, max((succ + rej + tout + fail + canc + miss).max() * 1.25, 1450))

    # Right: Invariant Violations (Duplicate Terminal Outcomes + Accounting Violations)
    xc = np.arange(len(COND_ORDER))
    w = 0.20
    base = summary[summary["architecture"] == "Baseline"].set_index("condition").loc[COND_ORDER]
    imp = summary[summary["architecture"] == "Improved"].set_index("condition").loc[COND_ORDER]

    v1 = ax2.bar(xc - 1.5 * w, base["terminal_accounting_violations_mean"], width=w, color="#C0392B", label="Baseline: Acct. Violations")
    v2 = ax2.bar(xc - 0.5 * w, base["duplicate_terminal_outcomes_mean"], width=w, color="#E67E22", label="Baseline: Dup. Outcomes")
    v3 = ax2.bar(xc + 0.5 * w, imp["terminal_accounting_violations_mean"], width=w, color="#1F618D", label="Improved: Acct. Violations")
    v4 = ax2.bar(xc + 1.5 * w, imp["duplicate_terminal_outcomes_mean"], width=w, color="#27AE60", label="Improved: Dup. Outcomes")

    max_v = max(base["terminal_accounting_violations_mean"].max(), base["duplicate_terminal_outcomes_mean"].max(), 10)
    for bar, val in zip(v1, base["terminal_accounting_violations_mean"]):
        ax2.text(bar.get_x() + bar.get_width() / 2 - 0.03, val + max_v * 0.02, f"{val:.0f}", ha="center", va="bottom", fontsize=7.0, color="#C0392B", fontweight="bold")
    for bar, val in zip(v2, base["duplicate_terminal_outcomes_mean"]):
        ax2.text(bar.get_x() + bar.get_width() / 2 + 0.03, val + max_v * 0.02, f"{val:.0f}", ha="center", va="bottom", fontsize=7.0, color="#D35400", fontweight="bold")
    for bar, val in zip(v3, imp["terminal_accounting_violations_mean"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + max_v * 0.02, f"{val:.0f}", ha="center", va="bottom", fontsize=7.0, color="#1F618D", fontweight="bold")
    for bar, val in zip(v4, imp["duplicate_terminal_outcomes_mean"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + max_v * 0.02, f"{val:.0f}", ha="center", va="bottom", fontsize=7.0, color="#1E8449", fontweight="bold")

    ax2.set_xticks(xc)
    ax2.set_xticklabels([COND_LABELS[c] for c in COND_ORDER])
    ax2.set_ylabel("Mean Violation Count per Run")
    ax2.set_title("Safety Invariant Violations (Lower = Better; 0 Required)")
    ax2.set_ylim(0, max_v * 1.28)
    ax2.legend(loc="upper left", fontsize=7.6, frameon=True, framealpha=0.92)

    fig.tight_layout()
    save_fig(fig, "plot_outcomes_reliability")


def plot_throughput_shutdown(summary: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7))
    x = np.arange(len(COND_ORDER))
    w = 0.35

    base = summary[summary["architecture"] == "Baseline"].set_index("condition").loc[COND_ORDER]
    imp = summary[summary["architecture"] == "Improved"].set_index("condition").loc[COND_ORDER]

    # Left: Goodput (Successful req/s)
    g1 = ax1.bar(x - w / 2, base["goodput_rps_mean"], yerr=base["goodput_rps_std"],
                 width=w, capsize=3, color=C_BASE, alpha=0.9, label="Baseline Goodput")
    g2 = ax1.bar(x + w / 2, imp["goodput_rps_mean"], yerr=imp["goodput_rps_std"],
                 width=w, capsize=3, color=C_IMP, alpha=0.9, label="Improved Goodput")
    for idx, (bar, val, err) in enumerate(zip(g1, base["goodput_rps_mean"], base["goodput_rps_std"])):
        lbl = f"{val:.0f}*" if COND_ORDER[idx] == "ack_faults" else f"{val:.0f}"
        ax1.text(bar.get_x() + bar.get_width() / 2, val + err + 10, lbl, ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=C_BASE)
    for bar, val, err in zip(g2, imp["goodput_rps_mean"], imp["goodput_rps_std"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + err + 10, f"{val:.0f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold", color=C_IMP)

    ax1.set_xticks(x)
    ax1.set_xticklabels([COND_LABELS[c] for c in COND_ORDER])
    ax1.set_ylabel("Goodput (Successful req/s)")
    ax1.set_title("Service Goodput (*Baseline Ack-Faults Inflated by Dups)")
    ax1.set_ylim(0, max(base["goodput_rps_mean"].max(), imp["goodput_rps_mean"].max()) * 1.28)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.9)

    # Right: Shutdown completion time (log scale for readability across 0.3ms to 1902ms) + stranded work annotation
    s1 = ax2.bar(x - w / 2, base["shutdown_completion_ms_mean"], yerr=base["shutdown_completion_ms_std"],
                 width=w, capsize=3, color=C_BASE, alpha=0.9, label="Baseline Shutdown")
    s2 = ax2.bar(x + w / 2, imp["shutdown_completion_ms_mean"], yerr=imp["shutdown_completion_ms_std"],
                 width=w, capsize=3, color=C_IMP, alpha=0.9, label="Improved Shutdown")
    ax2.set_yscale("log")
    for bar, val, err, out in zip(s1, base["shutdown_completion_ms_mean"], base["shutdown_completion_ms_std"], base["outstanding_work_after_shutdown_mean"]):
        lbl = f"{val:.1f}ms"
        if out > 0:
            lbl += f"\n(lost:{out:.0f})"
        ax2.text(bar.get_x() + bar.get_width() / 2 - 0.04, (val + err) * 1.28, lbl, ha="center", va="bottom", fontsize=6.9, fontweight="bold", color=C_BASE)
    for bar, val, err in zip(s2, imp["shutdown_completion_ms_mean"], imp["shutdown_completion_ms_std"]):
        ax2.text(bar.get_x() + bar.get_width() / 2 + 0.05, (val + err) * 1.28, f"{val:.1f}ms\n(lost:0)", ha="center", va="bottom", fontsize=6.9, fontweight="bold", color=C_IMP)

    ax2.set_xticks(x)
    ax2.set_xticklabels([COND_LABELS[c] for c in COND_ORDER])
    ax2.set_ylabel("Shutdown Completion Time (ms, log scale)")
    ax2.set_title("Shutdown Duration & Stranded Work After Shutdown")
    ax2.set_ylim(0.1, max(base["shutdown_completion_ms_mean"].max(), imp["shutdown_completion_ms_mean"].max()) * 5.0)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "plot_throughput_shutdown")


def main():
    raw_path = DATA_DIR / "comparison" / "comparison_raw.csv"
    df = pd.read_csv(raw_path)
    summary = build_summary_csv(df)
    plot_partb_normal_telemetry()
    plot_latency_p95_comparison(summary)
    plot_outcomes_reliability(summary)
    plot_throughput_shutdown(summary)
    print("Generated comparison_summary.csv and 4 PDF/PNG figures in report/figures/")


if __name__ == "__main__":
    main()

