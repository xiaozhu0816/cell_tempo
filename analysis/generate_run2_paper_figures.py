#!/usr/bin/env python3
"""
Generate publication-quality figures for HBMVEC Run2 external validation.

Uses the pre-computed predictions (run2_predictions.json) from validate_on_run2.py
and the corrected plate layout from the experimental protocol:

    Column 1 (a1, b1, c1): MOI = 5
    Column 2 (a2, b2, c2): MOI = 1
    Column 3 (a3, b3, c3): MOI = 0.1
    Column 4 (a4, b4, c4): Mock (PBS)

Experiment: HBMVEC passage 5, 45,000 cells/well, 93 scans, 30 min interval,
            imaged for ~48 h post infection.  TC-83 virus.

Figures produced:
    Fig 7A – Temporal classification curves (P(infected) vs hours) per MOI
    Fig 7B – Temporal regression curves (predicted vs actual hours) per MOI
    Fig 7C – Classification accuracy by time window (bar chart)
    Fig 7D – 12-well plate heatmap (P(infected) at selected time-points)
    
    Combined:  run2_fig7_external_validation.pdf / .png
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
CORRECTED_LAYOUT = {
    "MOI 5":   ["a1", "b1", "c1"],
    "MOI 1":   ["a2", "b2", "c2"],
    "MOI 0.1": ["a3", "b3", "c3"],
    "Mock":    ["a4", "b4", "c4"],
}

# True ground-truth label (infected / uninfected)
GT_LABEL = {}
for cond, wells in CORRECTED_LAYOUT.items():
    for w in wells:
        GT_LABEL[w] = 0 if cond == "Mock" else 1   # 1 = infected

COLORS = {
    "MOI 5":   "#E53935",   # red
    "MOI 1":   "#FF9800",   # orange
    "MOI 0.1": "#4CAF50",   # green
    "Mock":    "#2196F3",   # blue
}
COND_ORDER = ["Mock", "MOI 0.1", "MOI 1", "MOI 5"]

# Row / column helpers for plate grid
ROWS = ["a", "b", "c"]
COLS = [1, 2, 3, 4]
COL_COND = {1: "MOI 5", 2: "MOI 1", 3: "MOI 0.1", 4: "Mock"}

# ═══════════════════════════════════════════════════════════════════════════
# Load predictions
# ═══════════════════════════════════════════════════════════════════════════
def load_predictions(json_path: str) -> dict:
    with open(json_path) as f:
        raw = json.load(f)
    # Convert lists back to numpy
    for well, data in raw.items():
        for k in ("hours", "cls_prob_mean", "cls_prob_std",
                   "time_pred_mean", "time_pred_std"):
            if k in data:
                data[k] = np.array(data[k])
    return raw


def aggregate_by_condition(well_data: dict) -> dict:
    """Average across triplicate wells for each condition."""
    results = {}
    for cond in COND_ORDER:
        wells = CORRECTED_LAYOUT[cond]
        all_cls, all_time = [], []
        hours = None
        for w in wells:
            if w in well_data:
                wd = well_data[w]
                all_cls.append(wd["cls_prob_mean"])
                all_time.append(wd["time_pred_mean"])
                hours = wd["hours"]
        if all_cls:
            cls_arr = np.stack(all_cls)
            time_arr = np.stack(all_time)
            results[cond] = {
                "hours": hours,
                "cls_mean": cls_arr.mean(axis=0),
                "cls_std":  cls_arr.std(axis=0),
                "cls_wells": cls_arr,          # [3, T]
                "time_mean": time_arr.mean(axis=0),
                "time_std":  time_arr.std(axis=0),
                "n_wells": len(all_cls),
            }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Classification accuracy computation
# ═══════════════════════════════════════════════════════════════════════════
def compute_accuracy_by_window(well_data: dict, threshold: float = 0.5):
    """
    For each time window, compute binary classification accuracy
    using the known ground-truth labels (infected vs mock).
    
    Returns dict of {window_name: {accuracy, n_correct, n_total,
                                    tp, fp, tn, fn, sensitivity, specificity}}
    """
    windows = [
        ("0–6 h",   0,  6),
        ("6–12 h",  6, 12),
        ("12–24 h", 12, 24),
        ("24–46 h", 24, 46),
        ("Overall",  0, 46),
    ]
    results = {}
    for wname, t_lo, t_hi in windows:
        tp = fp = tn = fn = 0
        for well, wd in well_data.items():
            gt = GT_LABEL.get(well)
            if gt is None:
                continue
            h = wd["hours"]
            mask = (h >= t_lo) & (h <= t_hi)
            if not mask.any():
                continue
            probs = wd["cls_prob_mean"][mask]
            # Each time-point is a prediction
            preds = (probs >= threshold).astype(int)
            for p in preds:
                if gt == 1 and p == 1: tp += 1
                elif gt == 0 and p == 1: fp += 1
                elif gt == 0 and p == 0: tn += 1
                elif gt == 1 and p == 0: fn += 1

        n_total = tp + fp + tn + fn
        acc = (tp + tn) / n_total if n_total > 0 else 0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        results[wname] = {
            "accuracy": acc, "sensitivity": sens, "specificity": spec,
            "precision": prec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n_total": n_total,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Figure panels
# ═══════════════════════════════════════════════════════════════════════════

def panel_temporal_cls(ax, cond_data: dict):
    """Panel A: P(infected) temporal curves per MOI."""
    for cond in COND_ORDER:
        if cond not in cond_data:
            continue
        cd = cond_data[cond]
        color = COLORS[cond]
        h = cd["hours"]
        m = cd["cls_mean"]
        s = cd["cls_std"]
        ax.plot(h, m, '-o', markersize=3, linewidth=2, color=color, label=cond)
        ax.fill_between(h, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                        alpha=0.15, color=color)

    ax.axhline(0.5, ls='--', color='gray', lw=1, alpha=0.6)
    ax.set_xlabel("Hours post infection (hpi)", fontsize=11)
    ax.set_ylabel("P(infected)", fontsize=11)
    ax.set_title("(A) Classification probability over time",
                 fontweight="bold", loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="center right")
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0, 46])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)


def panel_temporal_reg(ax, cond_data: dict):
    """Panel B: Predicted time vs actual time per MOI."""
    for cond in COND_ORDER:
        if cond not in cond_data:
            continue
        cd = cond_data[cond]
        color = COLORS[cond]
        h = cd["hours"]
        m = cd["time_mean"]
        s = cd["time_std"]
        ax.plot(h, m, '-o', markersize=3, linewidth=2, color=color, label=cond)
        ax.fill_between(h, m - s, m + s, alpha=0.15, color=color)

    ax.plot([0, 46], [0, 46], 'k--', lw=1.5, alpha=0.4, label="Ideal (y = x)")
    ax.set_xlabel("Actual hours post infection (hpi)", fontsize=11)
    ax.set_ylabel("Predicted hours", fontsize=11)
    ax.set_title("(B) Time-since-infection regression",
                 fontweight="bold", loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim([0, 46])
    ax.set_ylim([0, 46])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)


def panel_accuracy_bars(ax, acc_data: dict):
    """Panel C: Classification accuracy (+ sensitivity / specificity) per time window."""
    windows = ["0–6 h", "6–12 h", "12–24 h", "24–46 h", "Overall"]
    x = np.arange(len(windows))
    width = 0.25

    acc_vals  = [acc_data[w]["accuracy"]    for w in windows]
    sens_vals = [acc_data[w]["sensitivity"] for w in windows]
    spec_vals = [acc_data[w]["specificity"] for w in windows]

    bars1 = ax.bar(x - width, acc_vals,  width, label="Accuracy",
                   color="#5C6BC0", edgecolor="black", lw=0.5)
    bars2 = ax.bar(x,         sens_vals, width, label="Sensitivity",
                   color="#EF5350", edgecolor="black", lw=0.5)
    bars3 = ax.bar(x + width, spec_vals, width, label="Specificity",
                   color="#66BB6A", edgecolor="black", lw=0.5)

    # Value labels on top
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(windows, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("(C) Classification performance by time window",
                 fontweight="bold", loc="left", fontsize=12)
    ax.set_ylim([0, 1.15])
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(labelsize=10)


def panel_plate_heatmap(ax, well_data: dict, time_hours: float = 24.0):
    """
    Panel D: 3×4 plate heatmap showing mean P(infected) at a specific time.
    """
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "pinf", ["#2196F3", "#FFFFFF", "#E53935"], N=256)

    plate = np.full((3, 4), np.nan)
    for ri, row in enumerate(ROWS):
        for ci, col in enumerate(COLS):
            well = f"{row}{col}"
            if well in well_data:
                wd = well_data[well]
                h = wd["hours"]
                idx = np.argmin(np.abs(h - time_hours))
                plate[ri, ci] = wd["cls_prob_mean"][idx]

    im = ax.imshow(plate, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"Col {c}\n({COL_COND[c]})" for c in COLS], fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"Row {r.upper()}" for r in ROWS], fontsize=10)

    # Annotate each cell
    for ri in range(3):
        for ci in range(4):
            v = plate[ri, ci]
            if not np.isnan(v):
                txt_color = "white" if v > 0.65 or v < 0.35 else "black"
                ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                        fontsize=12, fontweight="bold", color=txt_color)

    ax.set_title(f"(D) P(infected) at {time_hours:.0f} hpi — plate view",
                 fontweight="bold", loc="left", fontsize=12)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(infected)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)


def panel_per_well_grid(axes_grid, well_data: dict):
    """
    3×4 grid of per-well temporal curves, matching the physical plate layout.
    """
    for ri, row in enumerate(ROWS):
        for ci, col in enumerate(COLS):
            ax = axes_grid[ri, ci]
            well = f"{row}{col}"
            cond = COL_COND[col]
            color = COLORS[cond]

            if well in well_data:
                wd = well_data[well]
                h = wd["hours"]
                m = wd["cls_prob_mean"]
                s = wd["cls_prob_std"]
                ax.plot(h, m, '-', lw=1.5, color=color)
                ax.fill_between(h, np.clip(m - s, 0, 1),
                                np.clip(m + s, 0, 1),
                                alpha=0.2, color=color)
            ax.axhline(0.5, ls='--', color='gray', lw=0.7, alpha=0.5)
            ax.set_ylim([-0.05, 1.05])
            ax.set_xlim([0, 46])
            ax.grid(True, alpha=0.2)

            # Title: well + condition
            title_str = f"{well.upper()} — {cond}"
            ax.set_title(title_str, fontsize=8.5, fontweight="bold", color=color)

            if ri == 2:
                ax.set_xlabel("hpi", fontsize=8)
            else:
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel("P(inf)", fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)


# ═══════════════════════════════════════════════════════════════════════════
# Supplementary: Confusion matrix at a given time window
# ═══════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(acc_data: dict, output_dir: Path):
    """Generate a confusion matrix figure for the overall window."""
    overall = acc_data["Overall"]
    cm = np.array([[overall["tn"], overall["fp"]],
                   [overall["fn"], overall["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            txt_color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=txt_color)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred: Uninfected", "Pred: Infected"], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True: Mock", "True: Infected"], fontsize=9)
    ax.set_title("Overall Confusion Matrix (0–46 hpi)", fontweight="bold", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"run2_confusion_matrix.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"  Saved run2_confusion_matrix.pdf/.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main composite figure — Fig 7
# ═══════════════════════════════════════════════════════════════════════════
def generate_fig7(well_data: dict, output_dir: Path):
    """
    Fig 7: External validation on independent HBMVEC Run2 dataset
    
    Layout (2 rows):
        Top:    (A) temporal cls curves  |  (B) temporal regression curves
        Bottom: (C) accuracy bars        |  (D) plate heatmap at 24 hpi
    """
    cond_data = aggregate_by_condition(well_data)
    acc_data  = compute_accuracy_by_window(well_data)

    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.28,
                           left=0.07, right=0.95, top=0.94, bottom=0.06)

    # (A) Temporal classification curves
    ax_a = fig.add_subplot(gs[0, 0])
    panel_temporal_cls(ax_a, cond_data)

    # (B) Temporal regression
    ax_b = fig.add_subplot(gs[0, 1])
    panel_temporal_reg(ax_b, cond_data)

    # (C) Accuracy bars
    ax_c = fig.add_subplot(gs[1, 0])
    panel_accuracy_bars(ax_c, acc_data)

    # (D) Plate heatmap at 24 hpi
    ax_d = fig.add_subplot(gs[1, 1])
    panel_plate_heatmap(ax_d, well_data, time_hours=24.0)

    fig.suptitle("Figure 7. External Validation on Independent HBMVEC Run2 Dataset",
                 fontsize=14, fontweight="bold", y=0.98)

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"run2_fig7_external_validation.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"  Saved run2_fig7_external_validation.pdf/.png")
    plt.close(fig)


def generate_per_well_figure(well_data: dict, output_dir: Path):
    """
    Supplementary: 3×4 grid of per-well P(infected) curves matching plate layout.
    """
    fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey=True)
    panel_per_well_grid(axes, well_data)

    fig.suptitle("Supplementary: Per-Well Classification Curves — "
                 "HBMVEC Run2 External Validation",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"run2_supp_per_well.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"  Saved run2_supp_per_well.pdf/.png")
    plt.close(fig)


def generate_multi_timepoint_heatmap(well_data: dict, output_dir: Path):
    """
    Supplementary: Plate heatmaps at 4 time-points (6, 12, 24, 36 hpi).
    """
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "pinf", ["#2196F3", "#FFFFFF", "#E53935"], N=256)

    timepoints = [6, 12, 24, 36]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ti, (ax, tp) in enumerate(zip(axes, timepoints)):
        plate = np.full((3, 4), np.nan)
        for ri, row in enumerate(ROWS):
            for ci, col in enumerate(COLS):
                well = f"{row}{col}"
                if well in well_data:
                    wd = well_data[well]
                    h = wd["hours"]
                    idx = np.argmin(np.abs(h - tp))
                    plate[ri, ci] = wd["cls_prob_mean"][idx]

        im = ax.imshow(plate, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{COL_COND[c]}" for c in COLS], fontsize=8, rotation=20)
        ax.set_yticks(range(3))
        ax.set_yticklabels([r.upper() for r in ROWS], fontsize=10)

        for ri in range(3):
            for ci in range(4):
                v = plate[ri, ci]
                if not np.isnan(v):
                    txt_color = "white" if v > 0.65 or v < 0.35 else "black"
                    ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                            fontsize=10, fontweight="bold", color=txt_color)

        ax.set_title(f"{tp} hpi", fontweight="bold", fontsize=11)

    fig.suptitle("Supplementary: Plate Heatmaps at Multiple Time-Points",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.92, 0.92])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.65])
    plt.colorbar(im, cax=cbar_ax, label="P(infected)")

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"run2_supp_plate_heatmaps.{ext}",
                    dpi=300, bbox_inches="tight")
    print(f"  Saved run2_supp_plate_heatmaps.pdf/.png")
    plt.close(fig)


def print_summary_table(well_data: dict, acc_data: dict):
    """Print a formatted summary table to console."""
    print("\n" + "=" * 78)
    print("EXTERNAL VALIDATION RESULTS — HBMVEC Run2 (Corrected Plate Layout)")
    print("=" * 78)

    # Per-condition summary
    cond_data = aggregate_by_condition(well_data)
    print(f"\n{'Condition':10s} | {'Wells':12s} | {'GT Label':10s} | "
          f"{'Early P(inf)':>13s} | {'Late P(inf)':>12s} | {'Correct?':>9s}")
    print("-" * 78)
    for cond in COND_ORDER:
        wells_str = ",".join(CORRECTED_LAYOUT[cond])
        gt_str = "Mock" if cond == "Mock" else "Infected"
        cd = cond_data[cond]
        h = cd["hours"]
        early_mask = h <= 6
        late_mask  = h >= 24
        early_val = cd["cls_mean"][early_mask].mean() if early_mask.any() else float('nan')
        late_val  = cd["cls_mean"][late_mask].mean() if late_mask.any() else float('nan')
        # "Correct" if mock stays < 0.5 late, or infected goes > 0.5 late
        if cond == "Mock":
            correct = "✓" if late_val < 0.5 else "✗"
        else:
            correct = "✓" if late_val > 0.5 else "~"
        print(f"{cond:10s} | {wells_str:12s} | {gt_str:10s} | "
              f"{early_val:>13.4f} | {late_val:>12.4f} | {correct:>9s}")

    # Accuracy summary
    print(f"\n{'Window':12s} | {'Accuracy':>9s} | {'Sensitivity':>12s} | "
          f"{'Specificity':>12s} | {'F1':>6s} | {'N':>5s}")
    print("-" * 65)
    for wname in ["0–6 h", "6–12 h", "12–24 h", "24–46 h", "Overall"]:
        ad = acc_data[wname]
        print(f"{wname:12s} | {ad['accuracy']:>9.4f} | {ad['sensitivity']:>12.4f} | "
              f"{ad['specificity']:>12.4f} | {ad['f1']:>6.4f} | {ad['n_total']:>5d}")

    # Overall confusion
    ov = acc_data["Overall"]
    print(f"\nOverall confusion matrix (threshold=0.5):")
    print(f"  TP={ov['tp']}  FP={ov['fp']}  TN={ov['tn']}  FN={ov['fn']}")
    print(f"  Accuracy    = {ov['accuracy']:.4f}")
    print(f"  Sensitivity = {ov['sensitivity']:.4f}")
    print(f"  Specificity = {ov['specificity']:.4f}")
    print(f"  F1 score    = {ov['f1']:.4f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate paper figures for Run2 external validation")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to run2_predictions.json (auto-detected if omitted)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as predictions)")
    parser.add_argument("--heatmap-time", type=float, default=24.0,
                        help="Time-point for plate heatmap in Panel D (default: 24 hpi)")
    args = parser.parse_args()

    # Auto-detect predictions path
    if args.predictions:
        json_path = Path(args.predictions)
    else:
        # Try standard location
        base = Path(__file__).resolve().parent
        json_path = (base / "outputs" / "multitask_resnet50_crop5pct" /
                     "20260114-170730_5fold" / "Run2_Validation" /
                     "run2_predictions.json")
    if not json_path.exists():
        print(f"ERROR: Predictions file not found: {json_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else json_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING RUN2 PAPER FIGURES")
    print("=" * 70)
    print(f"  Predictions : {json_path}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Heatmap time: {args.heatmap_time} hpi")
    print()

    # Load data
    well_data = load_predictions(str(json_path))
    print(f"  Loaded {len(well_data)} wells")

    # Compute accuracy
    acc_data = compute_accuracy_by_window(well_data)

    # Print summary table
    print_summary_table(well_data, acc_data)

    # Generate figures
    print("Generating figures ...")
    generate_fig7(well_data, output_dir)
    generate_per_well_figure(well_data, output_dir)
    generate_multi_timepoint_heatmap(well_data, output_dir)
    plot_confusion_matrix(acc_data, output_dir)

    # Save accuracy data as JSON
    acc_json = output_dir / "run2_accuracy_metrics.json"
    with open(acc_json, "w") as f:
        json.dump(acc_data, f, indent=2)
    print(f"  Saved {acc_json}")

    print("\n" + "=" * 70)
    print("DONE — Paper figures generated successfully")
    print(f"  Main figure:   run2_fig7_external_validation.pdf")
    print(f"  Supplementary: run2_supp_per_well.pdf")
    print(f"                 run2_supp_plate_heatmaps.pdf")
    print(f"                 run2_confusion_matrix.pdf")
    print("=" * 70)


if __name__ == "__main__":
    main()
