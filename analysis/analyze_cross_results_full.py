from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

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
SPLITS = ["test_internal", "test_external"]
SPLIT_LABELS = {
    "test_internal": "Internal",
    "test_external": "External",
}
DIRECTIONS = ["ValRun2→ValRun3", "ValRun3→ValRun2"]
DIR_COLORS = {"ValRun2→ValRun3": "#1E88E5", "ValRun3→ValRun2": "#E53935"}
CLS_COLORS = {"MOI5": "#D32F2F", "MOI1": "#F57C00", "MOI0.1": "#1976D2", "Mock": "#388E3C"}
COND_ORDER = ["moi5", "moi1", "moi01", "mock"]
COND_DISPLAY = {"moi5": "MOI 5", "moi1": "MOI 1", "moi01": "MOI 0.1", "mock": "Mock"}
TIME_BIN_ORDER = ["0-6h", "6-12h", "12-18h", "18-24h", "24-30h", "30-36h", "36-42h", "42-48h"]


def load_json(p: Path) -> Dict:
    return json.loads(p.read_text(encoding="utf-8"))


def list_epochs(result_dir: Path, split: str = "test_external") -> List[int]:
    d = result_dir / split
    eps = sorted(int(x.name.split("_")[1]) for x in d.iterdir() if x.is_dir() and x.name.startswith("epoch_"))
    return eps


def load_metrics(result_dir: Path, split: str, epoch: int) -> Dict:
    return load_json(result_dir / split / f"epoch_{epoch:03d}" / "metrics.json")


def load_per_well(result_dir: Path, split: str, epoch: int) -> Dict:
    return load_json(result_dir / split / f"epoch_{epoch:03d}" / "per_well_metrics.json")


def load_predictions(result_dir: Path, split: str, epoch: int) -> Dict:
    p = result_dir / split / f"epoch_{epoch:03d}" / "predictions.npz"
    return dict(np.load(p, allow_pickle=True))


def load_eval_history(result_dir: Path) -> Dict:
    return load_json(result_dir / "eval_history.json")


def save_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


def save_table1(run_map: Dict[str, Path], epoch_map: Dict[str, int], odir: Path) -> None:
    header = ["Direction", "Split", "Accuracy", "F1 (macro)", "F1 (weighted)", "Precision", "Recall", "AUC (OVR)", "MAE (h)", "RMSE (h)", "R^2"]
    rows = []
    for direction, rdir in run_map.items():
        for split in SPLITS:
            m = load_metrics(rdir, split, epoch_map[direction])
            rows.append([
                direction,
                split,
                f"{m['cls_accuracy']*100:.2f}",
                f"{m['cls_f1_macro']:.4f}",
                f"{m['cls_f1_weighted']:.4f}",
                f"{m['cls_precision_macro']:.4f}",
                f"{m['cls_recall_macro']:.4f}",
                f"{m['cls_auc_macro']:.4f}",
                f"{m['reg_mae']:.3f}",
                f"{m['reg_rmse']:.3f}",
                f"{m['reg_r2']:.4f}",
            ])
    save_csv(odir / "table1_overall_metrics.csv", header, rows)


def save_table2(run_map: Dict[str, Path], epoch_map: Dict[str, int], odir: Path) -> None:
    header = ["Direction", "Split", "Class", "Precision", "Recall", "F1", "Support"]
    rows = []
    for direction, rdir in run_map.items():
        for split in SPLITS:
            m = load_metrics(rdir, split, epoch_map[direction])
            for cn in CLASS_NAMES:
                rows.append([
                    direction,
                    split,
                    cn,
                    f"{m.get(f'cls_{cn}_precision', 0):.4f}",
                    f"{m.get(f'cls_{cn}_recall', 0):.4f}",
                    f"{m.get(f'cls_{cn}_f1', 0):.4f}",
                    f"{int(m.get(f'cls_{cn}_support', 0))}",
                ])
    save_csv(odir / "table2_per_class_metrics.csv", header, rows)


def save_table3(run_map: Dict[str, Path], epoch_map: Dict[str, int], odir: Path) -> None:
    header = ["Direction", "Condition", "N", "Accuracy", "F1 macro", "MAE (h)", "RMSE (h)", "R^2"]
    rows = []
    for direction, rdir in run_map.items():
        pw = load_per_well(rdir, "test_external", epoch_map[direction])
        pc = pw["per_condition"]
        for cond in COND_ORDER:
            if cond not in pc:
                continue
            cd = pc[cond]
            rows.append([
                direction,
                COND_DISPLAY[cond],
                str(cd["n"]),
                f"{cd['accuracy']*100:.2f}",
                f"{cd['f1_macro']:.4f}",
                f"{cd['reg_mae']:.3f}",
                f"{cd['reg_rmse']:.3f}",
                f"{cd['reg_r2']:.4f}",
            ])
    save_csv(odir / "table3_per_condition.csv", header, rows)


def fig1_confusions(run_map: Dict[str, Path], epoch_map: Dict[str, int], odir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, direction in enumerate(DIRECTIONS):
        for j, split in enumerate(SPLITS):
            ax = axes[i, j]
            pred = load_predictions(run_map[direction], split, epoch_map[direction])
            cm = confusion_matrix(pred["labels"], pred["preds"], labels=list(range(len(CLASS_NAMES))))
            cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            im = ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
            ax.set_title(f"{direction} | {SPLIT_LABELS[split]}")
            ax.set_xticks(range(len(CLASS_NAMES)))
            ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
            ax.set_yticklabels(CLASS_NAMES)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for r in range(len(CLASS_NAMES)):
                for c in range(len(CLASS_NAMES)):
                    v = cmn[r, c]
                    ax.text(c, r, f"{v:.2f}\n({cm[r,c]})", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.5 else "black")
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Fraction")
    plt.suptitle("Fig1: Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    fig.savefig(odir / "fig1_confusion_matrices.png")
    fig.savefig(odir / "fig1_confusion_matrices.pdf")
    plt.close(fig)


def fig2_acc_f1(run_map: Dict[str, Path], odir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for direction in DIRECTIONS:
        eh = load_eval_history(run_map[direction])
        for split, style in [("test_internal", "-"), ("test_external", "--")]:
            xs = [r["epoch"] for r in eh[split]]
            acc = [r["metrics"]["cls_accuracy"] * 100 for r in eh[split]]
            f1 = [r["metrics"]["cls_f1_macro"] for r in eh[split]]
            label = f"{direction} | {SPLIT_LABELS[split]}"
            ax1.plot(xs, acc, style, color=DIR_COLORS[direction], lw=1.8, label=label)
            ax2.plot(xs, f1, style, color=DIR_COLORS[direction], lw=1.8, label=label)
    ax1.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Fig2: Accuracy Over Epochs")
    ax2.set(xlabel="Epoch", ylabel="F1 Macro", title="Fig2: F1 Over Epochs")
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(odir / "fig2_acc_f1_over_epochs.png")
    fig.savefig(odir / "fig2_acc_f1_over_epochs.pdf")
    plt.close(fig)


def fig3_regression(run_map: Dict[str, Path], odir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for direction in DIRECTIONS:
        eh = load_eval_history(run_map[direction])
        for split, style in [("test_internal", "-"), ("test_external", "--")]:
            xs = [r["epoch"] for r in eh[split]]
            mae = [r["metrics"]["reg_mae"] for r in eh[split]]
            r2 = [r["metrics"]["reg_r2"] for r in eh[split]]
            label = f"{direction} | {SPLIT_LABELS[split]}"
            ax1.plot(xs, mae, style, color=DIR_COLORS[direction], lw=1.8, label=label)
            ax2.plot(xs, r2, style, color=DIR_COLORS[direction], lw=1.8, label=label)
    ax1.set(xlabel="Epoch", ylabel="MAE (h)", title="Fig3: MAE Over Epochs")
    ax2.set(xlabel="Epoch", ylabel="R^2", title="Fig3: R^2 Over Epochs")
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(odir / "fig3_regression_over_epochs.png")
    fig.savefig(odir / "fig3_regression_over_epochs.pdf")
    plt.close(fig)


def fig4_per_class_f1(run_map: Dict[str, Path], epoch_map: Dict[str, int], odir: Path) -> None:
    x = np.arange(len(CLASS_NAMES))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, split in enumerate(SPLITS):
        ax = axes[i]
        vals1 = [load_metrics(run_map[DIRECTIONS[0]], split, epoch_map[DIRECTIONS[0]]).get(f"cls_{cn}_f1", 0) for cn in CLASS_NAMES]
        vals2 = [load_metrics(run_map[DIRECTIONS[1]], split, epoch_map[DIRECTIONS[1]]).get(f"cls_{cn}_f1", 0) for cn in CLASS_NAMES]
        ax.bar(x - width/2, vals1, width, label=DIRECTIONS[0], color=DIR_COLORS[DIRECTIONS[0]])
        ax.bar(x + width/2, vals2, width, label=DIRECTIONS[1], color=DIR_COLORS[DIRECTIONS[1]])
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Fig4: Per-class F1 ({SPLIT_LABELS[split]})")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(odir / "fig4_per_class_f1.png")
    fig.savefig(odir / "fig4_per_class_f1.pdf")
    plt.close(fig)


def fig5_time_bin_heatmap(run_map: Dict[str, Path], epoch_map: Dict[str, int], odir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, direction in enumerate(DIRECTIONS):
        ax = axes[i]
        pw = load_per_well(run_map[direction], "test_external", epoch_map[direction])
        pc = pw["per_condition"]
        mat = np.full((len(COND_ORDER), len(TIME_BIN_ORDER)), np.nan)
        for ri, cond in enumerate(COND_ORDER):
            bins = pc.get(cond, {}).get("time_bins", {})
            for ci, tb in enumerate(TIME_BIN_ORDER):
                if tb in bins:
                    mat[ri, ci] = bins[tb]["accuracy"] * 100
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(TIME_BIN_ORDER)))
        ax.set_xticklabels(TIME_BIN_ORDER, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(COND_ORDER)))
        ax.set_yticklabels([COND_DISPLAY[c] for c in COND_ORDER])
        ax.set_title(f"Fig5: External Time-bin Accuracy | {direction}")
        for r in range(len(COND_ORDER)):
            for c in range(len(TIME_BIN_ORDER)):
                if not np.isnan(mat[r, c]):
                    ax.text(c, r, f"{mat[r, c]:.0f}", ha="center", va="center", fontsize=7)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Accuracy (%)")
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    fig.savefig(odir / "fig5_time_bin_heatmap.png")
    fig.savefig(odir / "fig5_time_bin_heatmap.pdf")
    plt.close(fig)


def fig6_reg_scatter(run_map: Dict[str, Path], epoch_map: Dict[str, int], odir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for i, direction in enumerate(DIRECTIONS):
        ax = axes[i]
        pred = load_predictions(run_map[direction], "test_external", epoch_map[direction])
        tp = pred["time_preds"]
        tt = pred["time_targets"]
        labels = pred["labels"]
        for ci, cn in enumerate(CLASS_NAMES):
            mask = labels == ci
            ax.scatter(tt[mask], tp[mask], s=3, alpha=0.15, color=CLS_COLORS[cn], label=cn, rasterized=True)
        ax.plot([0, 48], [0, 48], "k--", lw=1, alpha=0.5)
        ax.set(xlim=[0, 48], ylim=[-2, 50], xlabel="True Time (h)", ylabel="Predicted Time (h)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Fig6: External Regression Scatter | {direction}")
        m = load_metrics(run_map[direction], "test_external", epoch_map[direction])
        ax.text(0.98, 0.05, f"MAE={m['reg_mae']:.2f}h\nR^2={m['reg_r2']:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8))
    axes[0].legend(markerscale=4, fontsize=8, loc="upper left")
    plt.tight_layout()
    fig.savefig(odir / "fig6_regression_scatter.png")
    fig.savefig(odir / "fig6_regression_scatter.pdf")
    plt.close(fig)


def fig7_well_accuracy(run_map: Dict[str, Path], epoch_map: Dict[str, int], odir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for i, direction in enumerate(DIRECTIONS):
        ax = axes[i]
        pw = load_per_well(run_map[direction], "test_external", epoch_map[direction])["per_well"]
        wells = sorted(pw.keys())
        vals = [pw[w]["accuracy"] * 100 for w in wells]
        ax.bar(range(len(wells)), vals, color=DIR_COLORS[direction], edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(wells)))
        ax.set_xticklabels(wells, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(f"Fig7: External Per-well Accuracy | {direction}")
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(odir / "fig7_per_well_accuracy.png")
    fig.savefig(odir / "fig7_per_well_accuracy.pdf")
    plt.close(fig)


def fig8_mae_by_condition(run_map: Dict[str, Path], epoch_map: Dict[str, int], odir: Path) -> None:
    x = np.arange(len(COND_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    vals1 = [load_per_well(run_map[DIRECTIONS[0]], "test_external", epoch_map[DIRECTIONS[0]])["per_condition"][c]["reg_mae"] for c in COND_ORDER]
    vals2 = [load_per_well(run_map[DIRECTIONS[1]], "test_external", epoch_map[DIRECTIONS[1]])["per_condition"][c]["reg_mae"] for c in COND_ORDER]
    ax.bar(x - width/2, vals1, width, label=DIRECTIONS[0], color=DIR_COLORS[DIRECTIONS[0]])
    ax.bar(x + width/2, vals2, width, label=DIRECTIONS[1], color=DIR_COLORS[DIRECTIONS[1]])
    ax.set_xticks(x)
    ax.set_xticklabels([COND_DISPLAY[c] for c in COND_ORDER])
    ax.set_ylabel("MAE (h)")
    ax.set_title("Fig8: External MAE by Condition")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(odir / "fig8_mae_by_condition.png")
    fig.savefig(odir / "fig8_mae_by_condition.pdf")
    plt.close(fig)


def fig9_all_metrics(run_map: Dict[str, Path], odir: Path) -> None:
    panels = [
        ("cls_accuracy", "Accuracy (%)", True),
        ("cls_f1_macro", "F1 Macro", False),
        ("cls_f1_weighted", "F1 Weighted", False),
        ("cls_precision_macro", "Precision Macro", False),
        ("cls_recall_macro", "Recall Macro", False),
        ("cls_auc_macro", "AUC Macro", False),
        ("reg_mae", "MAE (h)", False),
        ("reg_rmse", "RMSE (h)", False),
        ("reg_r2", "R^2", False),
    ]
    fig, axs = plt.subplots(3, 3, figsize=(17, 13))
    for ax, (mk, yl, p100) in zip(axs.flat, panels):
        for direction in DIRECTIONS:
            eh = load_eval_history(run_map[direction])
            for split, style in [("test_internal", "-"), ("test_external", "--")]:
                xs = [r["epoch"] for r in eh[split]]
                ys = [r["metrics"].get(mk, 0) for r in eh[split]]
                if p100:
                    ys = [v * 100 for v in ys]
                ax.plot(xs, ys, style, color=DIR_COLORS[direction], lw=1.4,
                        label=f"{direction}|{SPLIT_LABELS[split]}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl); ax.set_title(yl)
        ax.grid(True, alpha=0.3)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    plt.suptitle("Fig9: All Metrics Over Epochs", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(odir / "fig9_all_metrics_over_epochs.png")
    fig.savefig(odir / "fig9_all_metrics_over_epochs.pdf")
    plt.close(fig)


def fig10_per_class_f1_over_epochs(run_map: Dict[str, Path], odir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, direction in enumerate(DIRECTIONS):
        ax = axes[i]
        eh = load_eval_history(run_map[direction])["test_external"]
        xs = [r["epoch"] for r in eh]
        for cn in CLASS_NAMES:
            ys = [r["metrics"].get(f"cls_{cn}_f1", 0) for r in eh]
            ax.plot(xs, ys, "o-", color=CLS_COLORS[cn], lw=1.5, ms=3, label=cn)
        ax.set(xlabel="Epoch", ylabel="F1", title=f"Fig10: External Per-class F1 | {direction}")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(odir / "fig10_per_class_f1_over_epochs.png")
    fig.savefig(odir / "fig10_per_class_f1_over_epochs.pdf")
    plt.close(fig)


def figS1_roc(run_map: Dict[str, Path], epoch_map: Dict[str, int], odir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, direction in enumerate(DIRECTIONS):
        ax = axes[i]
        pred = load_predictions(run_map[direction], "test_external", epoch_map[direction])
        probs = pred["probs"]
        labels = pred["labels"]
        for ci, cn in enumerate(CLASS_NAMES):
            y_true = (labels == ci).astype(int)
            y_score = probs[:, ci]
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.5, color=CLS_COLORS[cn], label=f"{cn} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set(xlim=[0, 1], ylim=[0, 1.02], xlabel="FPR", ylabel="TPR", title=f"FigS1: External ROC | {direction}")
        ax.set_aspect("equal")
        ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    fig.savefig(odir / "figS1_roc_curves.png")
    fig.savefig(odir / "figS1_roc_curves.pdf")
    plt.close(fig)


def save_text_summary(run_map: Dict[str, Path], epoch_map: Dict[str, int], odir: Path) -> None:
    lines = []
    for direction in DIRECTIONS:
        m_int = load_metrics(run_map[direction], "test_internal", epoch_map[direction])
        m_ext = load_metrics(run_map[direction], "test_external", epoch_map[direction])
        lines.append(f"[{direction}] epoch={epoch_map[direction]}")
        lines.append(f"  Internal: Acc={m_int['cls_accuracy']*100:.2f}% F1m={m_int['cls_f1_macro']:.4f} MAE={m_int['reg_mae']:.3f}h R2={m_int['reg_r2']:.4f}")
        lines.append(f"  External: Acc={m_ext['cls_accuracy']*100:.2f}% F1m={m_ext['cls_f1_macro']:.4f} MAE={m_ext['reg_mae']:.3f}h R2={m_ext['reg_r2']:.4f}")
        lines.append("")
    ext1 = load_metrics(run_map[DIRECTIONS[0]], "test_external", epoch_map[DIRECTIONS[0]])
    ext2 = load_metrics(run_map[DIRECTIONS[1]], "test_external", epoch_map[DIRECTIONS[1]])
    lines.append("[External comparison]")
    lines.append(f"  ΔAcc (V2→V3 - V3→V2): {(ext1['cls_accuracy']-ext2['cls_accuracy'])*100:.2f}%")
    lines.append(f"  ΔF1m: {ext1['cls_f1_macro']-ext2['cls_f1_macro']:.4f}")
    lines.append(f"  ΔMAE: {ext1['reg_mae']-ext2['reg_mae']:.3f}h")
    lines.append(f"  ΔR2 : {ext1['reg_r2']-ext2['reg_r2']:.4f}")
    rec = DIRECTIONS[0] if (ext1['cls_f1_macro'] >= ext2['cls_f1_macro'] and ext1['reg_mae'] <= ext2['reg_mae']) else DIRECTIONS[1]
    lines.append(f"  Recommended direction: {rec}")
    (odir / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run2-dir", required=True, help="outputs/valrun2_train/<timestamp>")
    ap.add_argument("--run3-dir", required=True, help="outputs/valrun3_train/<timestamp>")
    ap.add_argument("--out-dir", default="paper_figures_cross_full")
    ap.add_argument("--epoch", type=int, default=None, help="Use fixed epoch for both dirs; default latest in each")
    args = ap.parse_args()

    run_map = {
        DIRECTIONS[0]: Path(args.run2_dir),
        DIRECTIONS[1]: Path(args.run3_dir),
    }
    for p in run_map.values():
        if not p.exists():
            raise FileNotFoundError(p)

    epoch_map = {}
    for direction, p in run_map.items():
        eps = list_epochs(p, "test_external")
        epoch_map[direction] = args.epoch if args.epoch is not None else eps[-1]

    odir = Path(args.out_dir)
    odir.mkdir(parents=True, exist_ok=True)

    save_table1(run_map, epoch_map, odir)
    save_table2(run_map, epoch_map, odir)
    save_table3(run_map, epoch_map, odir)

    fig1_confusions(run_map, epoch_map, odir)
    fig2_acc_f1(run_map, odir)
    fig3_regression(run_map, odir)
    fig4_per_class_f1(run_map, epoch_map, odir)
    fig5_time_bin_heatmap(run_map, epoch_map, odir)
    fig6_reg_scatter(run_map, epoch_map, odir)
    fig7_well_accuracy(run_map, epoch_map, odir)
    fig8_mae_by_condition(run_map, epoch_map, odir)
    fig9_all_metrics(run_map, odir)
    fig10_per_class_f1_over_epochs(run_map, odir)
    figS1_roc(run_map, epoch_map, odir)

    save_text_summary(run_map, epoch_map, odir)

    print("=" * 70)
    print("Cross full analysis done")
    print("Output:", odir)
    print("Epoch map:", epoch_map)
    print("=" * 70)


if __name__ == "__main__":
    main()
