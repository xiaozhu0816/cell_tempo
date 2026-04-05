#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunData:
    name: str
    root: Path
    summary: Dict
    eval_history: Dict


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_run_dir(exp_root: Path) -> Path:
    if not exp_root.exists():
        raise FileNotFoundError(f"Experiment folder not found: {exp_root}")
    candidates = sorted([p for p in exp_root.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No timestamp run folder under: {exp_root}")
    return candidates[-1]


def load_run(run_name: str, root: Path) -> RunData:
    return RunData(
        name=run_name,
        root=root,
        summary=load_json(root / "summary.json"),
        eval_history=load_json(root / "eval_history.json"),
    )


def get_external_series(rd: RunData) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float]]:
    recs = rd.eval_history["test_external"]
    epochs = [r["epoch"] for r in recs]
    acc = [r["metrics"].get("cls_accuracy", 0.0) for r in recs]
    f1m = [r["metrics"].get("cls_f1_macro", 0.0) for r in recs]
    auc = [r["metrics"].get("cls_auc_macro", 0.0) for r in recs]
    mae = [r["metrics"].get("reg_mae", 0.0) for r in recs]
    r2 = [r["metrics"].get("reg_r2", 0.0) for r in recs]
    return epochs, acc, f1m, auc, mae, r2


def best_epoch_by_metric(rd: RunData, metric: str, maximize: bool = True) -> Tuple[int, Dict]:
    recs = rd.eval_history["test_external"]
    key_fn = (lambda r: r["metrics"].get(metric, 0.0))
    best = max(recs, key=key_fn) if maximize else min(recs, key=key_fn)
    return int(best["epoch"]), best["metrics"]


def read_per_condition(run_root: Path, epoch: int) -> Dict:
    p = run_root / "test_external" / f"epoch_{epoch:03d}" / "per_well_metrics.json"
    data = load_json(p)
    return data["per_condition"]


def save_csv_table(path: Path, rows: List[Dict], headers: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def save_old_style_tables(run_a: RunData, run_b: RunData, out_dir: Path) -> None:
    experiments = [("ValRun2→ValRun3", run_a), ("ValRun3→ValRun2", run_b)]

    # Table 1: overall metrics (internal + external)
    table1_header = [
        "Experiment", "Split",
        "Accuracy", "F1 (macro)", "F1 (weighted)",
        "Precision", "Recall", "AUC (OVR)",
        "MAE (h)", "RMSE (h)", "R^2"
    ]
    t1_rows = []
    for exp_name, rd in experiments:
        for split in ["test_internal", "test_external"]:
            t1_rows.append({
                "Experiment": exp_name,
                "Split": split,
                "Accuracy": f"{rd.summary.get(f'{split}_acc', 0)*100:.2f}",
                "F1 (macro)": f"{rd.summary.get(f'{split}_f1_macro', 0):.4f}",
                "F1 (weighted)": f"{rd.summary.get(f'{split}_f1_wt', 0):.4f}",
                "Precision": f"{rd.summary.get(f'{split}_prec', 0):.4f}",
                "Recall": f"{rd.summary.get(f'{split}_rec', 0):.4f}",
                "AUC (OVR)": f"{rd.summary.get(f'{split}_auc', 0):.4f}",
                "MAE (h)": f"{rd.summary.get(f'{split}_mae', 0):.3f}",
                "RMSE (h)": f"{rd.summary.get(f'{split}_rmse', 0):.3f}",
                "R^2": f"{rd.summary.get(f'{split}_r2', 0):.4f}",
            })
    save_csv_table(out_dir / "table1_overall_metrics.csv", t1_rows, table1_header)

    with open(out_dir / "table1_overall_metrics.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Overall internal/external performance for cross-dataset directions.}\\n")
        f.write("\\label{tab:cross_overall}\\n")
        f.write("\\begin{tabular}{llccccccccc}\\n\\toprule\n")
        f.write("Experiment & Split & Acc(\\%) & F1m & F1w & Prec & Rec & AUC & MAE & RMSE & R$^2$ \\\\n\\midrule\n")
        for r in t1_rows:
            f.write(f"{r['Experiment']} & {r['Split']} & {r['Accuracy']} & {r['F1 (macro)']} & {r['F1 (weighted)']} & {r['Precision']} & {r['Recall']} & {r['AUC (OVR)']} & {r['MAE (h)']} & {r['RMSE (h)']} & {r['R^2']} \\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # Table 2: per-class external metrics
    class_names = ["MOI5", "MOI1", "MOI0.1", "Mock"]
    m_a = run_a.eval_history["test_external"][-1]["metrics"]
    m_b = run_b.eval_history["test_external"][-1]["metrics"]
    table2_header = [
        "Class",
        "ValRun2→ValRun3 P", "ValRun2→ValRun3 R", "ValRun2→ValRun3 F1",
        "ValRun3→ValRun2 P", "ValRun3→ValRun2 R", "ValRun3→ValRun2 F1",
    ]
    t2_rows = []
    for cn in class_names:
        t2_rows.append({
            "Class": cn,
            "ValRun2→ValRun3 P": f"{m_a.get(f'cls_{cn}_precision', 0):.4f}",
            "ValRun2→ValRun3 R": f"{m_a.get(f'cls_{cn}_recall', 0):.4f}",
            "ValRun2→ValRun3 F1": f"{m_a.get(f'cls_{cn}_f1', 0):.4f}",
            "ValRun3→ValRun2 P": f"{m_b.get(f'cls_{cn}_precision', 0):.4f}",
            "ValRun3→ValRun2 R": f"{m_b.get(f'cls_{cn}_recall', 0):.4f}",
            "ValRun3→ValRun2 F1": f"{m_b.get(f'cls_{cn}_f1', 0):.4f}",
        })
    save_csv_table(out_dir / "table2_per_class_metrics.csv", t2_rows, table2_header)

    with open(out_dir / "table2_per_class_metrics.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Per-class external precision/recall/F1 at final epoch.}\\n")
        f.write("\\label{tab:cross_perclass}\\n")
        f.write("\\begin{tabular}{lcccccc}\\n\\toprule\n")
        f.write("Class & V2→V3 P & V2→V3 R & V2→V3 F1 & V3→V2 P & V3→V2 R & V3→V2 F1 \\\\n\\midrule\n")
        for r in t2_rows:
            f.write(f"{r['Class']} & {r['ValRun2→ValRun3 P']} & {r['ValRun2→ValRun3 R']} & {r['ValRun2→ValRun3 F1']} & {r['ValRun3→ValRun2 P']} & {r['ValRun3→ValRun2 R']} & {r['ValRun3→ValRun2 F1']} \\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # Table 3: per-condition external metrics
    c_a = read_per_condition(run_a.root, epoch=100)
    c_b = read_per_condition(run_b.root, epoch=100)
    cond_order = ["moi5", "moi1", "moi01", "mock"]
    cond_disp = {"moi5": "MOI 5", "moi1": "MOI 1", "moi01": "MOI 0.1", "mock": "Mock"}
    table3_header = ["Condition", "Experiment", "N", "Accuracy", "F1 macro", "MAE (h)", "RMSE (h)", "R^2"]
    t3_rows = []
    for cond in cond_order:
        if cond in c_a:
            t3_rows.append({
                "Condition": cond_disp[cond],
                "Experiment": "ValRun2→ValRun3",
                "N": str(c_a[cond].get("n", 0)),
                "Accuracy": f"{c_a[cond].get('accuracy', 0)*100:.2f}",
                "F1 macro": f"{c_a[cond].get('f1_macro', 0):.4f}",
                "MAE (h)": f"{c_a[cond].get('reg_mae', 0):.3f}",
                "RMSE (h)": f"{c_a[cond].get('reg_rmse', 0):.3f}",
                "R^2": f"{c_a[cond].get('reg_r2', 0):.4f}",
            })
        if cond in c_b:
            t3_rows.append({
                "Condition": cond_disp[cond],
                "Experiment": "ValRun3→ValRun2",
                "N": str(c_b[cond].get("n", 0)),
                "Accuracy": f"{c_b[cond].get('accuracy', 0)*100:.2f}",
                "F1 macro": f"{c_b[cond].get('f1_macro', 0):.4f}",
                "MAE (h)": f"{c_b[cond].get('reg_mae', 0):.3f}",
                "RMSE (h)": f"{c_b[cond].get('reg_rmse', 0):.3f}",
                "R^2": f"{c_b[cond].get('reg_r2', 0):.4f}",
            })
    save_csv_table(out_dir / "table3_per_condition.csv", t3_rows, table3_header)


def figure_overall_comparison(run_a: RunData, run_b: RunData, out_dir: Path) -> None:
    labels = ["ValRun2→ValRun3", "ValRun3→ValRun2"]

    ext_acc = [run_a.summary["test_external_acc"], run_b.summary["test_external_acc"]]
    ext_f1 = [run_a.summary["test_external_f1_macro"], run_b.summary["test_external_f1_macro"]]
    ext_auc = [run_a.summary["test_external_auc"], run_b.summary["test_external_auc"]]
    ext_mae = [run_a.summary["test_external_mae"], run_b.summary["test_external_mae"]]

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    colors = ["#1E88E5", "#E53935"]

    axs[0, 0].bar(labels, [x * 100 for x in ext_acc], color=colors)
    axs[0, 0].set_title("External Accuracy (%)")
    axs[0, 0].set_ylim(0, 100)

    axs[0, 1].bar(labels, ext_f1, color=colors)
    axs[0, 1].set_title("External F1 Macro")
    axs[0, 1].set_ylim(0, 1)

    axs[1, 0].bar(labels, ext_auc, color=colors)
    axs[1, 0].set_title("External AUC Macro")
    axs[1, 0].set_ylim(0, 1)

    axs[1, 1].bar(labels, ext_mae, color=colors)
    axs[1, 1].set_title("External Regression MAE (h)")

    for ax in axs.flat:
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=10)

    fig.suptitle("Cross-Dataset External Generalization: Final Epoch Comparison", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_overall_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "fig1_overall_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_external_trajectories(run_a: RunData, run_b: RunData, out_dir: Path) -> None:
    e1, a1, f1_1, auc1, mae1, r2_1 = get_external_series(run_a)
    e2, a2, f1_2, auc2, mae2, r2_2 = get_external_series(run_b)

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    axs[0, 0].plot(e1, np.array(a1) * 100, "-o", ms=4, label="ValRun2→ValRun3")
    axs[0, 0].plot(e2, np.array(a2) * 100, "-o", ms=4, label="ValRun3→ValRun2")
    axs[0, 0].set_title("External Accuracy over Epochs")
    axs[0, 0].set_ylabel("Accuracy (%)")

    axs[0, 1].plot(e1, f1_1, "-o", ms=4, label="ValRun2→ValRun3")
    axs[0, 1].plot(e2, f1_2, "-o", ms=4, label="ValRun3→ValRun2")
    axs[0, 1].set_title("External F1 Macro over Epochs")

    axs[1, 0].plot(e1, mae1, "-o", ms=4, label="ValRun2→ValRun3")
    axs[1, 0].plot(e2, mae2, "-o", ms=4, label="ValRun3→ValRun2")
    axs[1, 0].set_title("External MAE over Epochs")
    axs[1, 0].set_ylabel("MAE (h)")

    axs[1, 1].plot(e1, r2_1, "-o", ms=4, label="ValRun2→ValRun3")
    axs[1, 1].plot(e2, r2_2, "-o", ms=4, label="ValRun3→ValRun2")
    axs[1, 1].set_title("External R² over Epochs")

    for ax in axs.flat:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle("Generalization Dynamics (External Test)", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_external_trajectories.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "fig2_external_trajectories.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_external_per_class_f1(run_a: RunData, run_b: RunData, out_dir: Path) -> None:
    keys = ["MOI5", "MOI1", "MOI0.1", "Mock"]

    m_a = run_a.eval_history["test_external"][-1]["metrics"]
    m_b = run_b.eval_history["test_external"][-1]["metrics"]

    vals_a = [m_a.get(f"cls_{k}_f1", 0.0) for k in keys]
    vals_b = [m_b.get(f"cls_{k}_f1", 0.0) for k in keys]

    x = np.arange(len(keys))
    w = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, vals_a, w, label="ValRun2→ValRun3", color="#1E88E5")
    ax.bar(x + w / 2, vals_b, w, label="ValRun3→ValRun2", color="#E53935")

    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("External Per-Class F1 (Final Epoch)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_external_per_class_f1.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "fig3_external_per_class_f1.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_condition_breakdown(run_a: RunData, run_b: RunData, out_dir: Path) -> None:
    cond_order = ["moi5", "moi1", "moi01", "mock"]

    c_a = read_per_condition(run_a.root, epoch=100)
    c_b = read_per_condition(run_b.root, epoch=100)

    acc_a = [c_a[c]["accuracy"] for c in cond_order]
    acc_b = [c_b[c]["accuracy"] for c in cond_order]
    mae_a = [c_a[c]["reg_mae"] for c in cond_order]
    mae_b = [c_b[c]["reg_mae"] for c in cond_order]

    x = np.arange(len(cond_order))
    w = 0.36

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].bar(x - w / 2, np.array(acc_a) * 100, w, label="ValRun2→ValRun3", color="#1E88E5")
    axs[0].bar(x + w / 2, np.array(acc_b) * 100, w, label="ValRun3→ValRun2", color="#E53935")
    axs[0].set_title("External Accuracy by Condition")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(["MOI5", "MOI1", "MOI0.1", "Mock"])
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].grid(True, axis="y", alpha=0.25)

    axs[1].bar(x - w / 2, mae_a, w, label="ValRun2→ValRun3", color="#1E88E5")
    axs[1].bar(x + w / 2, mae_b, w, label="ValRun3→ValRun2", color="#E53935")
    axs[1].set_title("External Regression MAE by Condition")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(["MOI5", "MOI1", "MOI0.1", "Mock"])
    axs[1].set_ylabel("MAE (h)")
    axs[1].grid(True, axis="y", alpha=0.25)

    for ax in axs:
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_condition_breakdown.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "fig4_condition_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate cross-dataset paper-style figures and analysis")
    ap.add_argument("--valrun2-root", type=str, default="outputs/valrun2_train")
    ap.add_argument("--valrun3-root", type=str, default="outputs/valrun3_train")
    ap.add_argument("--run-dir", type=str, default="", help="Optional explicit timestamp dir name")
    args = ap.parse_args()

    v2_root = Path(args.valrun2_root)
    v3_root = Path(args.valrun3_root)

    if args.run_dir:
        run2_dir = v2_root / args.run_dir
        run3_dir = v3_root / args.run_dir
    else:
        run2_dir = find_latest_run_dir(v2_root)
        run3_dir = find_latest_run_dir(v3_root)

    run2 = load_run("ValRun2→ValRun3", run2_dir)
    run3 = load_run("ValRun3→ValRun2", run3_dir)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("outputs") / "cross_dataset_paper_figs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_overall_comparison(run2, run3, out_dir)
    figure_external_trajectories(run2, run3, out_dir)
    figure_external_per_class_f1(run2, run3, out_dir)
    figure_condition_breakdown(run2, run3, out_dir)

    # Best-epoch analysis
    run2_best_f1_ep, run2_best_f1_m = best_epoch_by_metric(run2, "cls_f1_macro", maximize=True)
    run3_best_f1_ep, run3_best_f1_m = best_epoch_by_metric(run3, "cls_f1_macro", maximize=True)

    run2_best_mae_ep, run2_best_mae_m = best_epoch_by_metric(run2, "reg_mae", maximize=False)
    run3_best_mae_ep, run3_best_mae_m = best_epoch_by_metric(run3, "reg_mae", maximize=False)

    score2 = 0.6 * run2.summary["test_external_f1_macro"] + 0.4 * max(0.0, 1.0 - run2.summary["test_external_mae"] / 47.5)
    score3 = 0.6 * run3.summary["test_external_f1_macro"] + 0.4 * max(0.0, 1.0 - run3.summary["test_external_mae"] / 47.5)

    analysis = {
        "run2_final_external": {
            "acc": run2.summary["test_external_acc"],
            "f1_macro": run2.summary["test_external_f1_macro"],
            "auc": run2.summary["test_external_auc"],
            "mae": run2.summary["test_external_mae"],
            "rmse": run2.summary["test_external_rmse"],
            "r2": run2.summary["test_external_r2"],
        },
        "run3_final_external": {
            "acc": run3.summary["test_external_acc"],
            "f1_macro": run3.summary["test_external_f1_macro"],
            "auc": run3.summary["test_external_auc"],
            "mae": run3.summary["test_external_mae"],
            "rmse": run3.summary["test_external_rmse"],
            "r2": run3.summary["test_external_r2"],
        },
        "delta_run2_minus_run3": {
            "acc": run2.summary["test_external_acc"] - run3.summary["test_external_acc"],
            "f1_macro": run2.summary["test_external_f1_macro"] - run3.summary["test_external_f1_macro"],
            "auc": run2.summary["test_external_auc"] - run3.summary["test_external_auc"],
            "mae": run2.summary["test_external_mae"] - run3.summary["test_external_mae"],
            "rmse": run2.summary["test_external_rmse"] - run3.summary["test_external_rmse"],
            "r2": run2.summary["test_external_r2"] - run3.summary["test_external_r2"],
        },
        "best_external_f1": {
            "run2_epoch": run2_best_f1_ep,
            "run2_f1": run2_best_f1_m.get("cls_f1_macro", 0.0),
            "run3_epoch": run3_best_f1_ep,
            "run3_f1": run3_best_f1_m.get("cls_f1_macro", 0.0),
        },
        "best_external_mae": {
            "run2_epoch": run2_best_mae_ep,
            "run2_mae": run2_best_mae_m.get("reg_mae", 0.0),
            "run3_epoch": run3_best_mae_ep,
            "run3_mae": run3_best_mae_m.get("reg_mae", 0.0),
        },
        "external_composite_score": {
            "run2": score2,
            "run3": score3,
        },
        "recommended_direction": "ValRun2→ValRun3" if score2 >= score3 else "ValRun3→ValRun2"
    }

    with open(out_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    rows = [
        {
            "experiment": "ValRun2->ValRun3",
            "internal_acc": run2.summary["test_internal_acc"],
            "internal_f1": run2.summary["test_internal_f1_macro"],
            "internal_mae": run2.summary["test_internal_mae"],
            "external_acc": run2.summary["test_external_acc"],
            "external_f1": run2.summary["test_external_f1_macro"],
            "external_auc": run2.summary["test_external_auc"],
            "external_mae": run2.summary["test_external_mae"],
            "external_rmse": run2.summary["test_external_rmse"],
            "external_r2": run2.summary["test_external_r2"],
        },
        {
            "experiment": "ValRun3->ValRun2",
            "internal_acc": run3.summary["test_internal_acc"],
            "internal_f1": run3.summary["test_internal_f1_macro"],
            "internal_mae": run3.summary["test_internal_mae"],
            "external_acc": run3.summary["test_external_acc"],
            "external_f1": run3.summary["test_external_f1_macro"],
            "external_auc": run3.summary["test_external_auc"],
            "external_mae": run3.summary["test_external_mae"],
            "external_rmse": run3.summary["test_external_rmse"],
            "external_r2": run3.summary["test_external_r2"],
        },
    ]
    save_csv_table(
        out_dir / "comparison_table.csv",
        rows,
        headers=list(rows[0].keys()),
    )
    save_old_style_tables(run2, run3, out_dir)

    print("=" * 72)
    print("Cross-dataset analysis complete")
    print("Output folder:", out_dir)
    print("=" * 72)
    print("Recommended direction:", analysis["recommended_direction"])
    print("Final External (ValRun2→ValRun3): "
          f"Acc={run2.summary['test_external_acc']*100:.2f}% "
          f"F1={run2.summary['test_external_f1_macro']:.4f} "
          f"MAE={run2.summary['test_external_mae']:.3f}h")
    print("Final External (ValRun3→ValRun2): "
          f"Acc={run3.summary['test_external_acc']*100:.2f}% "
          f"F1={run3.summary['test_external_f1_macro']:.4f} "
          f"MAE={run3.summary['test_external_mae']:.3f}h")
    print("Best external F1 epochs:",
          f"run2 ep{run2_best_f1_ep} ({run2_best_f1_m.get('cls_f1_macro', 0):.4f}),",
          f"run3 ep{run3_best_f1_ep} ({run3_best_f1_m.get('cls_f1_macro', 0):.4f})")


if __name__ == "__main__":
    main()
