#!/usr/bin/env python3
"""
Earliest Divergence from Mock Analysis
=======================================
For each MOI condition, compute AUROC (condition vs mock) at each 3-hour bin.
Compare temporal vs single-frame models.
Outputs:
  - fig_earliest_divergence.pdf/png  (main Figure 3)
  - fig_confidence_trajectories.pdf/png (for Figure 4B)
  - earliest_divergence_summary.csv
"""

import json, sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
BASE = Path(r"\\medctr.ad.wfubmc.edu\dfs\gurcan_rsch$\scratch\WSI\zhengjie\CODE\cell_tempo")
OUT  = BASE / "paper" / "figures_revision"
OUT.mkdir(parents=True, exist_ok=True)

# 4-class models (preferred for per-MOI analysis since they preserve condition labels)
TEMPORAL_DIR = BASE / "outputs" / "rowsplit_4cls_temporal_v2" / "20260329-212856" / "test_internal" / "epoch_030"
SINGLE_DIR   = BASE / "outputs" / "rowsplit_4cls_v2" / "20260327-203141" / "test_internal" / "epoch_030"

# Also load binary models for comparison
BIN_TEMPORAL_DIR = BASE / "outputs" / "rowsplit_binary_temporal_v2" / "20260329-212856" / "test_internal" / "epoch_030"
BIN_SINGLE_DIR   = BASE / "outputs" / "rowsplit_binary_v2" / "20260329-010643" / "test_internal" / "epoch_030"

# ── Styling ────────────────────────────────────────────────────────
MOI_COLORS = {"moi5": "#D32F2F", "moi1": "#F57C00", "moi01": "#1976D2", "mock": "#388E3C"}
MOI_LABELS = {"moi5": "MOI 5", "moi1": "MOI 1", "moi01": "MOI 0.1", "mock": "Mock"}
INFECTED_MOIS = ["moi5", "moi1", "moi01"]

BIN_WIDTH = 3  # hours
AUROC_THRESHOLD = 0.90   # for "earliest separation"
ACC_THRESHOLD   = 0.85

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 10, "axes.linewidth": 1.0, "figure.dpi": 300
})


def load_samples(result_dir):
    """Load per_sample_results.json from a result directory."""
    p = Path(result_dir) / "per_sample_results.json"
    if not p.exists():
        print(f"[WARN] Not found: {p}")
        return []
    with open(p) as f:
        return json.load(f)


def bin_samples(samples, bin_width=3):
    """Bin samples by hours."""
    bins = {}
    for s in samples:
        h = s["hours"]
        bk = round((h // bin_width) * bin_width + bin_width / 2, 1)
        bins.setdefault(bk, []).append(s)
    return dict(sorted(bins.items()))


def compute_binary_auroc_per_condition(samples, moi_condition, bin_width=3):
    """
    For a specific MOI condition vs mock, compute AUROC at each time bin.
    Returns dict: {time_bin_center: {auroc, auprc, acc, n_moi, n_mock, sensitivity, specificity}}
    """
    # Filter to only this MOI + mock
    filtered = [s for s in samples if s["condition"] in (moi_condition, "mock")]
    if not filtered:
        return {}

    binned = bin_samples(filtered, bin_width)
    results = {}

    for bk, bsamps in binned.items():
        moi_samps  = [s for s in bsamps if s["condition"] == moi_condition]
        mock_samps = [s for s in bsamps if s["condition"] == "mock"]

        if len(moi_samps) < 5 or len(mock_samps) < 5:
            continue

        # Binary: MOI=1 (infected), Mock=0
        y_true = [1] * len(moi_samps) + [0] * len(mock_samps)

        # Probability of being infected (= 1 - prob_Mock)
        if "prob_Mock" in moi_samps[0]:
            y_score = [1.0 - s["prob_Mock"] for s in moi_samps + mock_samps]
        elif "prob_Infected" in moi_samps[0]:
            y_score = [s["prob_Infected"] for s in moi_samps + mock_samps]
        else:
            # For 4-class models: p(infected) = 1 - p(Mock)
            y_score = [1.0 - s.get("prob_Mock", 0.5) for s in moi_samps + mock_samps]

        y_pred = [1 if p > 0.5 else 0 for p in y_score]

        try:
            auroc = roc_auc_score(y_true, y_score)
        except ValueError:
            auroc = np.nan

        try:
            auprc = average_precision_score(y_true, y_score)
        except ValueError:
            auprc = np.nan

        acc = accuracy_score(y_true, y_pred)
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        sensitivity = tp / max(len(moi_samps), 1)
        specificity = tn / max(len(mock_samps), 1)

        results[bk] = {
            "auroc": auroc, "auprc": auprc, "accuracy": acc,
            "sensitivity": sensitivity, "specificity": specificity,
            "n_moi": len(moi_samps), "n_mock": len(mock_samps),
        }

    return results


def find_earliest_separation(per_bin_results, metric="auroc", threshold=0.90):
    """Find earliest time bin where metric exceeds threshold (sustained for 2+ consecutive bins)."""
    sorted_bins = sorted(per_bin_results.keys())
    for i, bk in enumerate(sorted_bins):
        val = per_bin_results[bk].get(metric, 0)
        if val >= threshold:
            # Check if sustained in next bin too (if available)
            if i + 1 < len(sorted_bins):
                next_val = per_bin_results[sorted_bins[i + 1]].get(metric, 0)
                if next_val >= threshold:
                    return bk
            else:
                return bk  # last bin, accept
    return None  # never reached threshold


def compute_confidence_trajectory(samples, bin_width=3):
    """
    Compute mean P(infected) per condition per time bin.
    Returns: {condition: {time_bin: {mean, sem, n}}}
    """
    by_cond = {}
    for s in samples:
        c = s["condition"]
        by_cond.setdefault(c, []).append(s)

    trajectories = {}
    for cond, csamps in by_cond.items():
        binned = bin_samples(csamps, bin_width)
        traj = {}
        for bk, bsamps in binned.items():
            if "prob_Mock" in bsamps[0]:
                probs = [1.0 - s["prob_Mock"] for s in bsamps]
            elif "prob_Infected" in bsamps[0]:
                probs = [s["prob_Infected"] for s in bsamps]
            else:
                probs = [1.0 - s.get("prob_Mock", 0.5) for s in bsamps]
            traj[bk] = {"mean": np.mean(probs), "sem": np.std(probs) / np.sqrt(len(probs)), "n": len(probs)}
        trajectories[cond] = dict(sorted(traj.items()))

    return trajectories


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("Loading prediction data...")
    temporal_samples = load_samples(TEMPORAL_DIR)
    single_samples   = load_samples(SINGLE_DIR)
    print(f"  Temporal 4cls: {len(temporal_samples)} samples")
    print(f"  Single   4cls: {len(single_samples)} samples")

    if not temporal_samples or not single_samples:
        # Fallback to different epoch directories
        print("[WARN] Trying alternative epoch directories...")
        alt_temporal = BASE / "outputs" / "rowsplit_4cls_temporal_v2" / "20260327-155528" / "test_internal" / "epoch_015"
        alt_single   = BASE / "outputs" / "rowsplit_4cls_v2" / "20260327-203141" / "test_internal" / "epoch_025"
        if not temporal_samples:
            temporal_samples = load_samples(alt_temporal)
            print(f"  Temporal 4cls (alt): {len(temporal_samples)} samples")
        if not single_samples:
            single_samples = load_samples(alt_single)
            print(f"  Single 4cls (alt): {len(single_samples)} samples")

    # ── Analysis 1: Earliest Divergence ────────────────────────────
    print("\n=== Earliest Divergence from Mock ===")
    summary_rows = []

    for moi in INFECTED_MOIS:
        for model_name, samples in [("Temporal", temporal_samples), ("Single-frame", single_samples)]:
            per_bin = compute_binary_auroc_per_condition(samples, moi, BIN_WIDTH)
            earliest_auroc = find_earliest_separation(per_bin, "auroc", AUROC_THRESHOLD)
            earliest_acc   = find_earliest_separation(per_bin, "accuracy", ACC_THRESHOLD)

            print(f"  {MOI_LABELS[moi]} ({model_name}):")
            print(f"    Earliest AUROC > {AUROC_THRESHOLD}: {earliest_auroc}h")
            print(f"    Earliest Acc   > {ACC_THRESHOLD}:  {earliest_acc}h")

            summary_rows.append({
                "MOI": MOI_LABELS[moi], "Model": model_name,
                "Earliest_AUROC_0.90": earliest_auroc,
                "Earliest_Acc_0.85": earliest_acc,
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT / "earliest_divergence_summary.csv", index=False)
    print(f"\nSaved: {OUT / 'earliest_divergence_summary.csv'}")

    # ── Figure 3: Earliest Divergence Figure ───────────────────────
    print("\nGenerating Figure 3: Earliest divergence...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax_idx, moi in enumerate(INFECTED_MOIS):
        ax = axes[ax_idx]

        for model_name, samples, ls, lw in [
            ("Temporal", temporal_samples, "-", 2.5),
            ("Single-frame", single_samples, "--", 2.0),
        ]:
            per_bin = compute_binary_auroc_per_condition(samples, moi, BIN_WIDTH)
            times = sorted(per_bin.keys())
            aurocs = [per_bin[t]["auroc"] for t in times]
            ax.plot(times, aurocs, ls, lw=lw, color=MOI_COLORS[moi],
                    label=model_name, alpha=0.9 if ls == "-" else 0.7)

        ax.axhline(AUROC_THRESHOLD, color="gray", ls=":", lw=1, alpha=0.5)
        ax.set_title(f"{MOI_LABELS[moi]} vs Mock", fontsize=12, fontweight="bold")
        ax.set_xlabel("Hours post-infection", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("AUROC (vs Mock)", fontsize=10)
        ax.set_ylim(0.4, 1.05)
        ax.set_xlim(0, 48)
        ax.legend(fontsize=8, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Dose-Dependent Temporal Divergence from Mock Controls", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(OUT / "fig_earliest_divergence.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "fig_earliest_divergence.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: fig_earliest_divergence.pdf/png")

    # ── Figure 3B: Summary bar chart ──────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(INFECTED_MOIS))
    width = 0.35

    temporal_earliest = []
    single_earliest = []
    for moi in INFECTED_MOIS:
        t_per = compute_binary_auroc_per_condition(temporal_samples, moi, BIN_WIDTH)
        s_per = compute_binary_auroc_per_condition(single_samples, moi, BIN_WIDTH)
        te = find_earliest_separation(t_per, "auroc", AUROC_THRESHOLD)
        se = find_earliest_separation(s_per, "auroc", AUROC_THRESHOLD)
        temporal_earliest.append(te if te is not None else 48)
        single_earliest.append(se if se is not None else 48)

    bars1 = ax2.bar(x - width/2, temporal_earliest, width, label="Temporal",
                    color=[MOI_COLORS[m] for m in INFECTED_MOIS], alpha=0.9, edgecolor="black", linewidth=0.5)
    bars2 = ax2.bar(x + width/2, single_earliest, width, label="Single-frame",
                    color=[MOI_COLORS[m] for m in INFECTED_MOIS], alpha=0.4, edgecolor="black", linewidth=0.5,
                    hatch="///")

    ax2.set_ylabel("Earliest Detection (hours)", fontsize=11)
    ax2.set_xlabel("Infection Dose", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([MOI_LABELS[m] for m in INFECTED_MOIS])
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 50)
    ax2.set_title("Earliest Detectable Divergence from Mock\n(AUROC > 0.90, sustained)", fontsize=11, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Add value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h < 48:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}h",
                     ha="center", va="bottom", fontsize=9)
        else:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, "N/A",
                     ha="center", va="bottom", fontsize=8, color="gray")

    plt.tight_layout()
    fig2.savefig(OUT / "fig_earliest_detection_bars.pdf", bbox_inches="tight", dpi=300)
    fig2.savefig(OUT / "fig_earliest_detection_bars.png", bbox_inches="tight", dpi=300)
    plt.close(fig2)
    print(f"  Saved: fig_earliest_detection_bars.pdf/png")

    # ── Analysis 2: Confidence Trajectories ────────────────────────
    print("\nGenerating confidence trajectory plots...")
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, samples, title in [
        (ax3a, temporal_samples, "Temporal Model"),
        (ax3b, single_samples, "Single-Frame Model"),
    ]:
        traj = compute_confidence_trajectory(samples, BIN_WIDTH)
        for cond in ["moi5", "moi1", "moi01", "mock"]:
            if cond not in traj:
                continue
            times = sorted(traj[cond].keys())
            means = [traj[cond][t]["mean"] for t in times]
            sems  = [traj[cond][t]["sem"] for t in times]
            ax.plot(times, means, "-", lw=2, color=MOI_COLORS[cond], label=MOI_LABELS[cond])
            ax.fill_between(times,
                            [m - s for m, s in zip(means, sems)],
                            [m + s for m, s in zip(means, sems)],
                            alpha=0.15, color=MOI_COLORS[cond])
        ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Hours post-infection", fontsize=10)
        ax.set_xlim(0, 48)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8, loc="center right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax3a.set_ylabel("P(Infected)", fontsize=11)
    fig3.suptitle("Predicted Infection Probability Over Time", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig3.savefig(OUT / "fig_confidence_trajectories.pdf", bbox_inches="tight", dpi=300)
    fig3.savefig(OUT / "fig_confidence_trajectories.png", bbox_inches="tight", dpi=300)
    plt.close(fig3)
    print(f"  Saved: fig_confidence_trajectories.pdf/png")

    # ── Detailed per-bin table ─────────────────────────────────────
    print("\nGenerating detailed per-bin AUROC table...")
    all_rows = []
    for moi in INFECTED_MOIS:
        for model_name, samples in [("Temporal", temporal_samples), ("Single-frame", single_samples)]:
            per_bin = compute_binary_auroc_per_condition(samples, moi, BIN_WIDTH)
            for t in sorted(per_bin.keys()):
                row = {"MOI": MOI_LABELS[moi], "Model": model_name, "Hour": t}
                row.update(per_bin[t])
                all_rows.append(row)

    df_detail = pd.DataFrame(all_rows)
    df_detail.to_csv(OUT / "per_bin_auroc_detail.csv", index=False)
    print(f"  Saved: per_bin_auroc_detail.csv")

    # ── Print summary table ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EARLIEST DIVERGENCE SUMMARY")
    print("=" * 70)
    print(f"{'MOI':<10} {'Model':<15} {'AUROC>0.90 (h)':<18} {'Acc>0.85 (h)':<15}")
    print("-" * 58)
    for _, row in df_summary.iterrows():
        auroc_h = f"{row['Earliest_AUROC_0.90']:.1f}" if row['Earliest_AUROC_0.90'] is not None else "N/A"
        acc_h   = f"{row['Earliest_Acc_0.85']:.1f}" if row['Earliest_Acc_0.85'] is not None else "N/A"
        print(f"{row['MOI']:<10} {row['Model']:<15} {auroc_h:<18} {acc_h:<15}")

    print("\nDone. All outputs in:", OUT)


if __name__ == "__main__":
    main()
