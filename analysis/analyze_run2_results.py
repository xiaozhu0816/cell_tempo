"""
analyze_run2_results.py
=======================
Generate publication-quality figures and tables from Run2 4-class
multitask training results.

Produces:
  1. Table 1  – Overall classification + regression metrics (3 test sets)
  2. Table 2  – Per-class precision / recall / F1 (3 test sets side-by-side)
  3. Table 3  – Per-condition (MOI level) metrics
  4. Fig 1    – Confusion matrices (test_a / test_b / test_c) at best epoch
  5. Fig 2    – Accuracy & F1 over training epochs (3 curves)
  6. Fig 3    – Regression MAE / R² over epochs
  7. Fig 4    – Per-class F1 bar chart (grouped by test set)
  8. Fig 5    – Time-bin accuracy heatmap (condition × time bin × test set)
  9. Fig 6    – Regression scatter (pred vs true) for each test set
  10. Fig 7   – Per-well accuracy bar chart
  11. Fig S1  – Per-class ROC curves (one panel per test set)

Usage:
    python analyze_run2_results.py [--result-dir outputs/run2_4class/20260302-145424]
                                   [--epoch 85]
                                   [--out-dir paper_figures_run2]
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.special import softmax
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                             classification_report)

# ── style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

CLASS_NAMES = ["MOI5", "MOI1", "MOI0.1", "Mock"]
TEST_SETS   = ["test_a", "test_b", "test_c"]
TEST_LABELS = {"test_a": "Test-A\n(Row A held-out)",
               "test_b": "Test-B\n(Row B external)",
               "test_c": "Test-C\n(Row C external)"}
TEST_LABELS_SHORT = {"test_a": "Test-A (Row A)",
                     "test_b": "Test-B (Row B)",
                     "test_c": "Test-C (Row C)"}
COLORS  = {"test_a": "#E53935", "test_b": "#1E88E5", "test_c": "#43A047"}
CLS_COLORS = {"MOI5": "#D32F2F", "MOI1": "#F57C00",
              "MOI0.1": "#1976D2", "Mock": "#388E3C"}

COND_ORDER = ["moi5", "moi1", "moi01", "mock"]
COND_DISPLAY = {"moi5": "MOI 5", "moi1": "MOI 1",
                "moi01": "MOI 0.1", "mock": "Mock"}
TIME_BIN_ORDER = ["0-6h", "6-12h", "12-18h", "18-24h",
                  "24-30h", "30-36h", "36-42h", "42-48h"]


# ═══════════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════════
def load_metrics(rdir, tset, epoch):
    p = Path(rdir) / tset / f"epoch_{epoch:03d}" / "metrics.json"
    return json.loads(p.read_text())


def load_per_well(rdir, tset, epoch):
    p = Path(rdir) / tset / f"epoch_{epoch:03d}" / "per_well_metrics.json"
    return json.loads(p.read_text())


def load_predictions(rdir, tset, epoch):
    p = Path(rdir) / tset / f"epoch_{epoch:03d}" / "predictions.npz"
    return dict(np.load(p, allow_pickle=True))


def available_epochs(rdir, tset="test_a"):
    d = Path(rdir) / tset
    eps = sorted(int(x.name.split("_")[1]) for x in d.iterdir()
                 if x.is_dir() and x.name.startswith("epoch_"))
    return eps


def load_metrics_over_epochs(rdir):
    """Load metrics for all epochs and all test sets."""
    eps = available_epochs(rdir)
    data = {ts: [] for ts in TEST_SETS}
    for ep in eps:
        for ts in TEST_SETS:
            try:
                m = load_metrics(rdir, ts, ep)
                data[ts].append({"epoch": ep, **m})
            except FileNotFoundError:
                pass
    return data


# ═══════════════════════════════════════════════════════════════════════════
# Table 1 – Overall metrics
# ═══════════════════════════════════════════════════════════════════════════
def make_table1(rdir, epoch, odir):
    """Overall classification + regression metrics for 3 test sets."""
    rows = []
    header = ("Test Set", "Accuracy", "F1 (macro)", "F1 (weighted)",
              "Precision", "Recall", "AUC (OVR)",
              "MAE (h)", "RMSE (h)", "R\u00b2")
    for ts in TEST_SETS:
        m = load_metrics(rdir, ts, epoch)
        rows.append((
            TEST_LABELS_SHORT[ts],
            f"{m['cls_accuracy']*100:.1f}",
            f"{m['cls_f1_macro']:.4f}",
            f"{m['cls_f1_weighted']:.4f}",
            f"{m['cls_precision_macro']:.4f}",
            f"{m['cls_recall_macro']:.4f}",
            f"{m['cls_auc_macro']:.4f}",
            f"{m['reg_mae']:.2f}",
            f"{m['reg_rmse']:.2f}",
            f"{m['reg_r2']:.4f}",
        ))
    # Save as CSV
    with open(odir / "table1_overall_metrics.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")

    # Save as LaTeX
    with open(odir / "table1_overall_metrics.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Overall classification and regression performance "
                "on three test sets at epoch " + str(epoch) + ".}\n")
        f.write("\\label{tab:overall}\n")
        cols = "l" + "c" * (len(header) - 1)
        f.write(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"  Table 1 saved: table1_overall_metrics.csv / .tex")
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# Table 2 – Per-class metrics
# ═══════════════════════════════════════════════════════════════════════════
def make_table2(rdir, epoch, odir):
    """Per-class P/R/F1 for each test set."""
    header = ["Class"]
    for ts in TEST_SETS:
        short = TEST_LABELS_SHORT[ts]
        header += [f"{short} P", f"{short} R", f"{short} F1"]

    rows = []
    for cn in CLASS_NAMES:
        row = [cn]
        for ts in TEST_SETS:
            m = load_metrics(rdir, ts, epoch)
            row.append(f"{m.get(f'cls_{cn}_precision', 0):.4f}")
            row.append(f"{m.get(f'cls_{cn}_recall', 0):.4f}")
            row.append(f"{m.get(f'cls_{cn}_f1', 0):.4f}")
        rows.append(row)

    with open(odir / "table2_per_class_metrics.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")

    with open(odir / "table2_per_class_metrics.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Per-class precision, recall and F1 across test sets "
                "(epoch " + str(epoch) + ").}\n")
        f.write("\\label{tab:perclass}\n")
        nc = len(header)
        colspec = "l" + "c" * (nc - 1)
        f.write(f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n")
        # Multi-row header
        f.write("& " + " & ".join(
            [f"\\multicolumn{{3}}{{c}}{{{TEST_LABELS_SHORT[ts]}}}"
             for ts in TEST_SETS]) + " \\\\\n")
        f.write("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n")
        f.write("Class & P & R & F1 & P & R & F1 & P & R & F1 \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"  Table 2 saved: table2_per_class_metrics.csv / .tex")


# ═══════════════════════════════════════════════════════════════════════════
# Table 3 – Per-condition (column) metrics
# ═══════════════════════════════════════════════════════════════════════════
def make_table3(rdir, epoch, odir):
    header = ["Condition", "Test Set", "N", "Accuracy",
              "F1 macro", "MAE (h)", "RMSE (h)", "R\u00b2"]
    rows = []
    for cond in COND_ORDER:
        for ts in TEST_SETS:
            pw = load_per_well(rdir, ts, epoch)
            cd = pw["per_condition"].get(cond, {})
            if not cd:
                continue
            rows.append([
                COND_DISPLAY[cond],
                TEST_LABELS_SHORT[ts],
                str(cd["n"]),
                f"{cd['accuracy']*100:.1f}",
                f"{cd['f1_macro']:.4f}",
                f"{cd['reg_mae']:.2f}",
                f"{cd['reg_rmse']:.2f}",
                f"{cd['reg_r2']:.4f}",
            ])

    with open(odir / "table3_per_condition.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"  Table 3 saved: table3_per_condition.csv")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1 – Confusion matrices (3 panels)
# ═══════════════════════════════════════════════════════════════════════════
def fig1_confusion_matrices(rdir, epoch, odir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, ts in zip(axes, TEST_SETS):
        pred = load_predictions(rdir, ts, epoch)
        cm = confusion_matrix(pred["labels"], pred["preds"],
                              labels=list(range(len(CLASS_NAMES))))
        # Normalise by row (true class)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues",
                       interpolation="nearest")
        ax.set_title(TEST_LABELS_SHORT[ts], fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val = cm_norm[i, j]
                cnt = cm[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}\n({cnt})", ha="center",
                        va="center", fontsize=9, color=color)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Fraction")

    plt.suptitle(f"Normalized Confusion Matrices (Epoch {epoch})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    fig.savefig(odir / "fig1_confusion_matrices.png")
    fig.savefig(odir / "fig1_confusion_matrices.pdf")
    plt.close(fig)
    print("  Fig 1 saved: fig1_confusion_matrices")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2 – Accuracy & F1 over epochs
# ═══════════════════════════════════════════════════════════════════════════
def fig2_acc_f1_over_epochs(rdir, odir):
    data = load_metrics_over_epochs(rdir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ts in TEST_SETS:
        if not data[ts]:
            continue
        eps = [d["epoch"] for d in data[ts]]
        acc = [d["cls_accuracy"] * 100 for d in data[ts]]
        f1  = [d["cls_f1_macro"] for d in data[ts]]
        ax1.plot(eps, acc, "o-", color=COLORS[ts],
                 label=TEST_LABELS_SHORT[ts], lw=1.8, ms=4)
        ax2.plot(eps, f1, "o-", color=COLORS[ts],
                 label=TEST_LABELS_SHORT[ts], lw=1.8, ms=4)

    ax1.set(xlabel="Epoch", ylabel="Accuracy (%)",
            title="Classification Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set(xlabel="Epoch", ylabel="F1 (macro)",
            title="Macro F1 Score")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(odir / "fig2_acc_f1_over_epochs.png")
    fig.savefig(odir / "fig2_acc_f1_over_epochs.pdf")
    plt.close(fig)
    print("  Fig 2 saved: fig2_acc_f1_over_epochs")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3 – Regression MAE & R² over epochs
# ═══════════════════════════════════════════════════════════════════════════
def fig3_regression_over_epochs(rdir, odir):
    data = load_metrics_over_epochs(rdir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ts in TEST_SETS:
        if not data[ts]:
            continue
        eps = [d["epoch"] for d in data[ts]]
        mae = [d["reg_mae"] for d in data[ts]]
        r2  = [d["reg_r2"] for d in data[ts]]
        ax1.plot(eps, mae, "o-", color=COLORS[ts],
                 label=TEST_LABELS_SHORT[ts], lw=1.8, ms=4)
        ax2.plot(eps, r2, "o-", color=COLORS[ts],
                 label=TEST_LABELS_SHORT[ts], lw=1.8, ms=4)

    ax1.set(xlabel="Epoch", ylabel="MAE (hours)",
            title="Regression MAE")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set(xlabel="Epoch", ylabel="R\u00b2",
            title="Regression R\u00b2")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(odir / "fig3_regression_over_epochs.png")
    fig.savefig(odir / "fig3_regression_over_epochs.pdf")
    plt.close(fig)
    print("  Fig 3 saved: fig3_regression_over_epochs")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4 – Per-class F1 bar chart (grouped by test set)
# ═══════════════════════════════════════════════════════════════════════════
def fig4_per_class_f1_bars(rdir, epoch, odir):
    fig, ax = plt.subplots(figsize=(10, 5))
    n_cls = len(CLASS_NAMES)
    n_ts  = len(TEST_SETS)
    width = 0.22
    x = np.arange(n_cls)

    for i, ts in enumerate(TEST_SETS):
        m = load_metrics(rdir, ts, epoch)
        vals = [m.get(f"cls_{cn}_f1", 0) for cn in CLASS_NAMES]
        bars = ax.bar(x + i * width, vals, width,
                      label=TEST_LABELS_SHORT[ts], color=COLORS[ts],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-Class F1 Score (Epoch {epoch})", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(odir / "fig4_per_class_f1.png")
    fig.savefig(odir / "fig4_per_class_f1.pdf")
    plt.close(fig)
    print("  Fig 4 saved: fig4_per_class_f1")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 5 – Time-bin accuracy heatmap
# ═══════════════════════════════════════════════════════════════════════════
def fig5_time_bin_heatmap(rdir, epoch, odir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, ts in zip(axes, TEST_SETS):
        pw = load_per_well(rdir, ts, epoch)
        pc = pw["per_condition"]

        # Build matrix: rows = conditions, cols = time bins
        mat = np.full((len(COND_ORDER), len(TIME_BIN_ORDER)), np.nan)
        for ci, cond in enumerate(COND_ORDER):
            bins = pc.get(cond, {}).get("time_bins", {})
            for ti, tb in enumerate(TIME_BIN_ORDER):
                if tb in bins:
                    mat[ci, ti] = bins[tb]["accuracy"] * 100

        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100,
                       aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(TIME_BIN_ORDER)))
        ax.set_xticklabels(TIME_BIN_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(COND_ORDER)))
        ax.set_yticklabels([COND_DISPLAY[c] for c in COND_ORDER])
        ax.set_title(TEST_LABELS_SHORT[ts], fontweight="bold")
        ax.set_xlabel("Time Bin")

        for ci in range(len(COND_ORDER)):
            for ti in range(len(TIME_BIN_ORDER)):
                v = mat[ci, ti]
                if not np.isnan(v):
                    color = "white" if v < 40 else "black"
                    ax.text(ti, ci, f"{v:.0f}", ha="center", va="center",
                            fontsize=8, color=color)

    plt.suptitle(f"Classification Accuracy by Condition and Time Window (Epoch {epoch})",
                 fontsize=13, fontweight="bold", y=1.02)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Accuracy (%)")
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    fig.savefig(odir / "fig5_time_bin_heatmap.png")
    fig.savefig(odir / "fig5_time_bin_heatmap.pdf")
    plt.close(fig)
    print("  Fig 5 saved: fig5_time_bin_heatmap")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 6 – Regression scatter (pred vs true)
# ═══════════════════════════════════════════════════════════════════════════
def fig6_regression_scatter(rdir, epoch, odir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for ax, ts in zip(axes, TEST_SETS):
        pred = load_predictions(rdir, ts, epoch)
        tp = pred["time_preds"]
        tt = pred["time_targets"]
        labels = pred["labels"]

        for ci, cn in enumerate(CLASS_NAMES):
            mask = labels == ci
            ax.scatter(tt[mask], tp[mask], s=3, alpha=0.15,
                       color=CLS_COLORS[cn], label=cn, rasterized=True)

        lims = [0, 48]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5)
        ax.set(xlim=lims, ylim=[-2, 50],
               xlabel="True Time (h)", ylabel="Predicted Time (h)")
        ax.set_title(TEST_LABELS_SHORT[ts], fontweight="bold")
        ax.legend(markerscale=4, fontsize=8, loc="upper left")
        ax.set_aspect("equal", adjustable="box")

        # Add MAE / R² annotation
        m = load_metrics(rdir, ts, epoch)
        ax.text(0.98, 0.05,
                f"MAE={m['reg_mae']:.2f}h\nR\u00b2={m['reg_r2']:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3",
                                       facecolor="white", alpha=0.8))

    plt.suptitle(f"Time-Since-Infection Regression (Epoch {epoch})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(odir / "fig6_regression_scatter.png")
    fig.savefig(odir / "fig6_regression_scatter.pdf")
    plt.close(fig)
    print("  Fig 6 saved: fig6_regression_scatter")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 7 – Per-well accuracy bar chart
# ═══════════════════════════════════════════════════════════════════════════
def fig7_per_well_bars(rdir, epoch, odir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, ts in zip(axes, TEST_SETS):
        pw = load_per_well(rdir, ts, epoch)["per_well"]
        wells = sorted(pw.keys())
        accs = [pw[w]["accuracy"] * 100 for w in wells]
        conds = [pw[w]["condition"] for w in wells]
        colors = [CLS_COLORS.get(
            {"moi5": "MOI5", "moi1": "MOI1", "moi01": "MOI0.1",
             "mock": "Mock"}.get(c, ""), "#999") for c in conds]

        bars = ax.bar(range(len(wells)), accs, color=colors,
                      edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(wells)))
        ax.set_xticklabels([f"{w}\n({COND_DISPLAY.get(c, c)})"
                            for w, c in zip(wells, conds)],
                           fontsize=8, rotation=0)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(TEST_LABELS_SHORT[ts], fontweight="bold")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3)

        for bar, v in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    plt.suptitle(f"Per-Well Classification Accuracy (Epoch {epoch})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(odir / "fig7_per_well_accuracy.png")
    fig.savefig(odir / "fig7_per_well_accuracy.pdf")
    plt.close(fig)
    print("  Fig 7 saved: fig7_per_well_accuracy")


# ═══════════════════════════════════════════════════════════════════════════
# Fig S1 – Per-class ROC curves
# ═══════════════════════════════════════════════════════════════════════════
def figs1_roc_curves(rdir, epoch, odir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for ax, ts in zip(axes, TEST_SETS):
        pred = load_predictions(rdir, ts, epoch)
        probs  = pred["probs"]
        labels = pred["labels"]
        n_cls  = len(CLASS_NAMES)

        for ci, cn in enumerate(CLASS_NAMES):
            y_true = (labels == ci).astype(int)
            y_score = probs[:, ci]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=CLS_COLORS[cn], lw=1.5,
                    label=f"{cn} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set(xlim=[0, 1], ylim=[0, 1.02],
               xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax.set_title(TEST_LABELS_SHORT[ts], fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_aspect("equal")

    plt.suptitle(f"Per-Class ROC Curves — One-vs-Rest (Epoch {epoch})",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(odir / "figS1_roc_curves.png")
    fig.savefig(odir / "figS1_roc_curves.pdf")
    plt.close(fig)
    print("  Fig S1 saved: figS1_roc_curves")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 8 – Regression MAE by condition (grouped bar)
# ═══════════════════════════════════════════════════════════════════════════
def fig8_mae_by_condition(rdir, epoch, odir):
    fig, ax = plt.subplots(figsize=(10, 5))
    n_cond = len(COND_ORDER)
    n_ts   = len(TEST_SETS)
    width  = 0.22
    x = np.arange(n_cond)

    for i, ts in enumerate(TEST_SETS):
        pw = load_per_well(rdir, ts, epoch)
        pc = pw["per_condition"]
        vals = [pc.get(c, {}).get("reg_mae", 0) for c in COND_ORDER]
        bars = ax.bar(x + i * width, vals, width,
                      label=TEST_LABELS_SHORT[ts], color=COLORS[ts],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([COND_DISPLAY[c] for c in COND_ORDER], fontsize=11)
    ax.set_ylabel("MAE (hours)")
    ax.set_title(f"Regression MAE by MOI Condition (Epoch {epoch})",
                 fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(odir / "fig8_mae_by_condition.png")
    fig.savefig(odir / "fig8_mae_by_condition.pdf")
    plt.close(fig)
    print("  Fig 8 saved: fig8_mae_by_condition")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 9 – Comprehensive 3×3 metrics over epochs
# ═══════════════════════════════════════════════════════════════════════════
def fig9_all_metrics_over_epochs(rdir, odir):
    data = load_metrics_over_epochs(rdir)
    panels = [
        ("cls_accuracy",        "Accuracy (%)",        True),
        ("cls_f1_macro",        "F1 Macro",            False),
        ("cls_f1_weighted",     "F1 Weighted",         False),
        ("cls_precision_macro", "Precision Macro",     False),
        ("cls_recall_macro",    "Recall Macro",        False),
        ("cls_auc_macro",       "AUC Macro",           False),
        ("reg_mae",             "MAE (h)",             False),
        ("reg_rmse",            "RMSE (h)",            False),
        ("reg_r2",              "R\u00b2",             False),
    ]

    fig, axs = plt.subplots(3, 3, figsize=(17, 13))
    for ax, (mk, yl, s100) in zip(axs.flat, panels):
        for ts in TEST_SETS:
            if not data[ts]:
                continue
            eps = [d["epoch"] for d in data[ts]]
            ys  = [d.get(mk, 0) for d in data[ts]]
            if s100:
                ys = [y * 100 for y in ys]
            ax.plot(eps, ys, "o-", color=COLORS[ts],
                    label=TEST_LABELS_SHORT[ts], lw=1.5, ms=3)
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl); ax.set_title(yl)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle("All Evaluation Metrics Over Training Epochs",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(odir / "fig9_all_metrics_over_epochs.png")
    fig.savefig(odir / "fig9_all_metrics_over_epochs.pdf")
    plt.close(fig)
    print("  Fig 9 saved: fig9_all_metrics_over_epochs")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 10 – Per-class F1 over epochs (per test set)
# ═══════════════════════════════════════════════════════════════════════════
def fig10_per_class_f1_over_epochs(rdir, odir):
    data = load_metrics_over_epochs(rdir)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, ts in zip(axes, TEST_SETS):
        if not data[ts]:
            continue
        eps = [d["epoch"] for d in data[ts]]
        for cn in CLASS_NAMES:
            key = f"cls_{cn}_f1"
            ys = [d.get(key, 0) for d in data[ts]]
            ax.plot(eps, ys, "o-", color=CLS_COLORS[cn],
                    label=cn, lw=1.5, ms=3)
        ax.set(xlabel="Epoch", ylabel="F1",
               title=TEST_LABELS_SHORT[ts])
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle("Per-Class F1 Over Training Epochs",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(odir / "fig10_per_class_f1_over_epochs.png")
    fig.savefig(odir / "fig10_per_class_f1_over_epochs.pdf")
    plt.close(fig)
    print("  Fig 10 saved: fig10_per_class_f1_over_epochs")


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir",
                    default="outputs/run2_4class/20260302-145424")
    ap.add_argument("--epoch", type=int, default=None,
                    help="Epoch to use for per-epoch figures. "
                         "Default: latest available.")
    ap.add_argument("--out-dir", default="paper_figures_run2")
    args = ap.parse_args()

    rdir = Path(args.result_dir)
    odir = Path(args.out_dir)
    odir.mkdir(parents=True, exist_ok=True)

    eps = available_epochs(rdir)
    epoch = args.epoch if args.epoch else eps[-1]
    print(f"Result dir : {rdir}")
    print(f"Output dir : {odir}")
    print(f"Epoch      : {epoch}  (available: {eps[0]}..{eps[-1]})")
    print(f"{'='*60}")

    # Tables
    print("\n[Tables]")
    make_table1(rdir, epoch, odir)
    make_table2(rdir, epoch, odir)
    make_table3(rdir, epoch, odir)

    # Figures
    print("\n[Figures]")
    fig1_confusion_matrices(rdir, epoch, odir)
    fig2_acc_f1_over_epochs(rdir, odir)
    fig3_regression_over_epochs(rdir, odir)
    fig4_per_class_f1_bars(rdir, epoch, odir)
    fig5_time_bin_heatmap(rdir, epoch, odir)
    fig6_regression_scatter(rdir, epoch, odir)
    fig7_per_well_bars(rdir, epoch, odir)
    fig8_mae_by_condition(rdir, epoch, odir)
    fig9_all_metrics_over_epochs(rdir, odir)
    fig10_per_class_f1_over_epochs(rdir, odir)

    # Supplementary
    print("\n[Supplementary]")
    figs1_roc_curves(rdir, epoch, odir)

    print(f"\n{'='*60}")
    print(f"All done! {odir.resolve()}")


if __name__ == "__main__":
    main()
