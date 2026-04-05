#!/usr/bin/env python3
"""
Generate Supplementary Figures for Cell Reports Revision
=========================================================
Generates:
  S1. Training curves (loss + accuracy + regression over epochs)
  S2. Per-well accuracy heatmap at selected timepoints
  S3. Per-class precision/recall breakdown at early/mid/late phases
  S4. Dataset statistics (samples per condition x time)
  S5. Confusion matrices per time phase (early/mid/late) for both models
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "paper" / "figures_revision" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 10,
    "axes.linewidth": 1.0,
    "figure.dpi": 300,
})

# ──────────────────────────────────────────────
# S1: Training curves from eval_history.json
# ──────────────────────────────────────────────
def generate_training_curves():
    """Plot training curves for temporal 4cls and SF 4cls models."""
    models = {
        "Temporal 4-class": BASE / "outputs" / "rowsplit_4cls_temporal_v2" / "20260329-212856" / "eval_history.json",
        "Single-frame 4-class": BASE / "outputs" / "rowsplit_4cls_v2" / "20260327-203141" / "eval_history.json",
        "Temporal binary": BASE / "outputs" / "rowsplit_binary_temporal_v2" / "20260329-212856" / "eval_history.json",
        "Single-frame binary": BASE / "outputs" / "rowsplit_binary_v2" / "20260329-010643" / "eval_history.json",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    metrics_to_plot = [
        ("cls_accuracy", "Classification Accuracy"),
        ("cls_f1_macro", "Macro F1-Score"),
        ("reg_mae", "Regression MAE (h)"),
        ("reg_r2", "Regression R²"),
        ("total_loss", "Total Loss"),
        ("bin_accuracy", "Binary Accuracy (derived)"),
    ]
    colors = {
        "Temporal 4-class": "#2166ac",
        "Single-frame 4-class": "#b2182b",
        "Temporal binary": "#4393c3",
        "Single-frame binary": "#d6604d",
    }

    for ax_idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
        ax = axes.flat[ax_idx]
        for model_name, path in models.items():
            if not path.exists():
                continue
            with open(path) as f:
                history = json.load(f)
            if "test_internal" not in history:
                continue
            entries = history["test_internal"]
            epochs = [e["epoch"] for e in entries]
            vals = [e["metrics"].get(metric_key) for e in entries]
            if any(v is None for v in vals):
                continue
            ax.plot(epochs, vals, "-o", ms=3, lw=1.5, label=model_name,
                    color=colors.get(model_name, "gray"))
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Figure S1. Training curves across model configurations",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figS1_training_curves.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "figS1_training_curves.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: figS1_training_curves.pdf")


# ──────────────────────────────────────────────
# S2: Dataset statistics
# ──────────────────────────────────────────────
def generate_dataset_stats():
    """Generate dataset composition table/figure."""
    path = BASE / "outputs" / "rowsplit_4cls_temporal_v2" / "20260329-212856" / "test_internal" / "epoch_030" / "per_sample_results.json"
    if not path.exists():
        print("Skipping dataset stats — data not found")
        return

    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Samples per condition
    cond_counts = df["condition"].value_counts()
    print(f"\nDataset composition (test set):")
    print(cond_counts)

    # Samples per condition x time bin
    df["time_bin"] = (df["hours"] // 3 * 3 + 1.5).round(1)
    pivot = df.groupby(["condition", "time_bin"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([{"moi5":"MOI 5","moi1":"MOI 1","moi0.1":"MOI 0.1","mock":"Mock"}.get(c, c) for c in pivot.index])
    time_labels = [f"{t:.0f}h" for t in pivot.columns]
    ax.set_xticks(range(len(time_labels)))
    ax.set_xticklabels(time_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Hours post-infection")
    ax.set_title("Figure S2. Test set sample counts per condition and time bin",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="N samples", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "figS2_dataset_stats.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "figS2_dataset_stats.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: figS2_dataset_stats.pdf")

    # Also save as LaTeX table
    total_df = df.groupby("condition").agg(
        n_samples=("index", "count"),
        n_wells=("well", "nunique"),
        n_positions=("position", "nunique"),
    ).reset_index()
    total_df.to_csv(OUT / "dataset_composition.csv", index=False)
    print(f"Saved: dataset_composition.csv")


# ──────────────────────────────────────────────
# S3: Late-stage collapse analysis (detailed)
# ──────────────────────────────────────────────
def generate_late_stage_analysis():
    """Detailed per-condition AUROC with error bars showing late-stage collapse."""
    from sklearn.metrics import roc_auc_score

    models = {
        "Temporal": BASE / "outputs" / "rowsplit_4cls_temporal_v2" / "20260329-212856" / "test_internal" / "epoch_030" / "per_sample_results.json",
        "Single-frame": BASE / "outputs" / "rowsplit_4cls_v2" / "20260327-203141" / "test_internal" / "epoch_030" / "per_sample_results.json",
    }

    conditions = ["moi5", "moi1", "moi0.1"]
    cond_labels = {"moi5": "MOI 5", "moi1": "MOI 1", "moi0.1": "MOI 0.1"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for ax, cond in zip(axes, conditions):
        for model_name, path in models.items():
            if not path.exists():
                continue
            with open(path) as f:
                data = json.load(f)
            df = pd.DataFrame(data)

            # Filter to this condition + mock
            mask = df["condition"].isin([cond, "mock"])
            sub = df[mask].copy()
            sub["is_infected"] = (sub["condition"] != "mock").astype(int)
            sub["p_infected"] = 1.0 - sub["prob_Mock"]
            sub["time_bin"] = (sub["hours"] // 3 * 3 + 1.5).round(1)

            bins = sorted(sub["time_bin"].unique())
            aurocs = []
            for tb in bins:
                bsub = sub[sub["time_bin"] == tb]
                if bsub["is_infected"].nunique() < 2 or len(bsub) < 10:
                    aurocs.append(np.nan)
                    continue
                aurocs.append(roc_auc_score(bsub["is_infected"], bsub["p_infected"]))

            ls = "-" if model_name == "Temporal" else "--"
            c = "#2166ac" if model_name == "Temporal" else "#b2182b"
            ax.plot(bins, aurocs, ls, lw=2, color=c, marker="o", ms=3, label=model_name)

        ax.axhline(0.90, color="gray", ls=":", alpha=0.4)
        ax.axhline(0.50, color="red", ls=":", alpha=0.3, label="Chance")
        ax.fill_betweenx([0.4, 1.05], 40, 48, alpha=0.08, color="red")
        ax.set_title(f"{cond_labels[cond]} vs Mock", fontsize=12, fontweight="bold")
        ax.set_xlabel("Hours post-infection")
        ax.set_xlim(0, 48)
        ax.set_ylim(0.4, 1.05)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("AUROC (vs Mock)")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Figure S3. Late-stage collapse: per-condition AUROC over time",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "figS3_late_stage_collapse.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "figS3_late_stage_collapse.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: figS3_late_stage_collapse.pdf")


# ──────────────────────────────────────────────
# S4: Per-class precision/recall at different phases
# ──────────────────────────────────────────────
def generate_per_class_phase_metrics():
    """Per-class precision/recall for early/mid/late phases."""
    from sklearn.metrics import classification_report

    path = BASE / "outputs" / "rowsplit_4cls_temporal_v2" / "20260329-212856" / "test_internal" / "epoch_030" / "per_sample_results.json"
    if not path.exists():
        return

    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    phases = [("Early (0–12h)", 0, 12), ("Mid (12–24h)", 12, 24), ("Late (24–48h)", 24, 48)]
    class_names = ["MOI5", "MOI1", "MOI0.1", "Mock"]

    rows = []
    for phase_name, t0, t1 in phases:
        sub = df[(df["hours"] >= t0) & (df["hours"] < t1)]
        for cls_idx, cls_name in enumerate(class_names):
            cls_mask = sub["true_label"] == cls_idx
            n = cls_mask.sum()
            if n == 0:
                continue
            correct = (sub["pred_label"] == cls_idx) & cls_mask
            precision = correct.sum() / max((sub["pred_label"] == cls_idx).sum(), 1)
            recall = correct.sum() / max(n, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            rows.append({
                "Phase": phase_name, "Class": cls_name,
                "N": int(n), "Precision": round(precision, 3),
                "Recall": round(recall, 3), "F1": round(f1, 3),
            })

    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(OUT / "per_class_phase_metrics.csv", index=False)
    print(f"Saved: per_class_phase_metrics.csv")

    # Generate LaTeX table
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Phase} & \textbf{Class} & \textbf{N} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\",
        r"\midrule",
    ]
    for _, row in df_metrics.iterrows():
        lines.append(f"{row['Phase']} & {row['Class']} & {row['N']} & {row['Precision']:.3f} & {row['Recall']:.3f} & {row['F1']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "tableS1_per_class_phase_metrics.tex").write_text("\n".join(lines))
    print(f"Saved: tableS1_per_class_phase_metrics.tex")


if __name__ == "__main__":
    print("Generating supplementary figures for Cell Reports revision...\n")
    generate_training_curves()
    generate_dataset_stats()
    generate_late_stage_analysis()
    generate_per_class_phase_metrics()
    print("\nAll supplementary figures generated.")
