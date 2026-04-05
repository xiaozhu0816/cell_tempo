"""
val_train.py  --  Cross-dataset training for Validation Run experiments.

Strategy
--------
  Dataset A (all 12 wells): 70% train / 30% internal-test (by position)
  Dataset B (all 12 wells): 100% external-test

Run both directions:
  python val_train.py --config configs/valrun2_train.yaml   # Train P4, Ext P5
  python val_train.py --config configs/valrun3_train.yaml   # Train P5, Ext P4
"""
from __future__ import annotations

import argparse, json, math, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn
from scipy.special import softmax
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── local imports ────────────────────────────────────────────────────────────
from datasets.run2_dataset import (build_cross_dataset,
                                   build_multi_train_external_dataset,
                                   build_row_split_dataset)
from models import build_multitask_model
from utils import (AverageMeter, multiclass_metrics,
                   build_transforms, get_logger, load_config, set_seed)


# ═══════════════════════════════════════════════════════════════════════════
# helpers (same as run2_train.py)
# ═══════════════════════════════════════════════════════════════════════════
def _meta_to_list(meta_batch) -> List[Dict[str, Any]]:
    if isinstance(meta_batch, list):
        return meta_batch
    keys = list(meta_batch.keys())
    n = len(meta_batch[keys[0]])
    return [{k: (meta_batch[k][i].item()
                 if isinstance(meta_batch[k][i], torch.Tensor)
                 else meta_batch[k][i])
             for k in keys} for i in range(n)]


def _time_targets(meta_list, clamp, device):
    lo, hi = clamp
    t = [max(lo, min(hi, float(m.get("hours", 0)))) for m in meta_list]
    return torch.tensor(t, dtype=torch.float32).unsqueeze(1).to(device)


def _reg_metrics(p, t):
    d = p - t
    mae = float(np.abs(d).mean())
    mse = float((d ** 2).mean())
    rmse = float(np.sqrt(mse))
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    return dict(mae=mae, rmse=rmse, mse=mse, r2=r2)


def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer, np.floating)):
            out[k] = float(v)
        elif isinstance(v, dict):
            out[k] = _jsonable(v)
        else:
            out[k] = v
    return out


def compute_binary_from_4cls(raw, num_classes=4):
    """Derive binary (Infected vs Mock) metrics from 4-class predictions.
    Classes 0..num_classes-2 → Infected (0), class num_classes-1 → Mock (1).
    Returns dict with binary metrics + binary labels/preds arrays."""
    labels_4 = raw["labels"]
    preds_4 = raw["preds"]
    probs_4 = raw["probs"]

    # Binary: mock (last class) = 1, all others = 0
    mock_idx = num_classes - 1
    bin_labels = (labels_4 == mock_idx).astype(int)
    bin_preds = (preds_4 == mock_idx).astype(int)
    # Binary prob of "mock" = prob of last class; prob of "infected" = sum of rest
    bin_prob_mock = probs_4[:, mock_idx]

    acc = accuracy_score(bin_labels, bin_preds)
    f1 = f1_score(bin_labels, bin_preds, average="macro", zero_division=0)
    prec = precision_score(bin_labels, bin_preds, average="macro", zero_division=0)
    rec = recall_score(bin_labels, bin_preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(bin_labels, bin_prob_mock)
    except ValueError:
        auc = 0.0

    return dict(
        bin_accuracy=acc, bin_f1_macro=f1, bin_precision_macro=prec,
        bin_recall_macro=rec, bin_auc=auc,
        bin_labels=bin_labels, bin_preds=bin_preds, bin_prob_mock=bin_prob_mock)


# ═══════════════════════════════════════════════════════════════════════════
# train one epoch
# ═══════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, cls_crit, reg_crit, optim, scaler,
                    dev, clamp, cw, rw, amp, gclip, epoch):
    model.train()
    L = AverageMeter("L"); Lc = AverageMeter("Lc"); Lr = AverageMeter("Lr")
    correct = total = 0

    for imgs, labs, meta in tqdm(loader, desc=f"Train E{epoch}", leave=False):
        imgs  = imgs.to(dev, non_blocking=True)
        ctgt  = labs.long().to(dev, non_blocking=True)
        ttgt  = _time_targets(_meta_to_list(meta), clamp, dev)

        optim.zero_grad()
        if amp and dev.type == "cuda":
            with torch.cuda.amp.autocast():
                clog, tp = model(imgs)
                cl = cls_crit(clog, ctgt); rl = reg_crit(tp, ttgt)
                loss = cw * cl + rw * rl
        else:
            clog, tp = model(imgs)
            cl = cls_crit(clog, ctgt); rl = reg_crit(tp, ttgt)
            loss = cw * cl + rw * rl

        scaler.scale(loss).backward()
        if gclip:
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), gclip)
        scaler.step(optim); scaler.update()

        bs = imgs.size(0)
        L.update(loss.item(), bs); Lc.update(cl.item(), bs); Lr.update(rl.item(), bs)
        correct += (clog.argmax(1) == ctgt).sum().item(); total += bs

    return dict(total_loss=L.avg, cls_loss=Lc.avg, reg_loss=Lr.avg,
                train_acc=correct / max(total, 1))


# ═══════════════════════════════════════════════════════════════════════════
# evaluate
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, cls_crit, reg_crit, dev, clamp,
             cnames, cw, rw, desc="Eval"):
    model.eval()
    L = AverageMeter("L"); Lc = AverageMeter("Lc"); Lr = AverageMeter("Lr")
    all_log, all_lab, all_tp, all_tt = [], [], [], []
    all_meta: List[Dict] = []

    for imgs, labs, meta in tqdm(loader, desc=desc, leave=False):
        imgs = imgs.to(dev, non_blocking=True)
        ctgt = labs.long().to(dev, non_blocking=True)
        ml   = _meta_to_list(meta)
        ttgt = _time_targets(ml, clamp, dev)

        clog, tp = model(imgs)
        cl = cls_crit(clog, ctgt); rl = reg_crit(tp, ttgt)
        loss = cw * cl + rw * rl

        bs = imgs.size(0)
        L.update(loss.item(), bs); Lc.update(cl.item(), bs); Lr.update(rl.item(), bs)
        all_log.append(clog.cpu().numpy())
        all_lab.append(ctgt.cpu().numpy())
        all_tp.append(tp.cpu().numpy())
        all_tt.append(ttgt.cpu().numpy())
        all_meta.extend(ml)

    logits = np.concatenate(all_log)
    labels = np.concatenate(all_lab)
    probs  = softmax(logits, axis=1)
    preds  = probs.argmax(1)
    t_p    = np.concatenate(all_tp).squeeze(-1)
    t_t    = np.concatenate(all_tt).squeeze(-1)

    cm = multiclass_metrics(probs, labels, cnames)
    rm = _reg_metrics(t_p, t_t)
    rs = max(0, 1 - rm["mae"] / clamp[1])
    combined = 0.6 * cm["f1_macro"] + 0.4 * rs

    met = dict(total_loss=L.avg, cls_loss=Lc.avg, reg_loss=Lr.avg, combined=combined)
    met.update({f"cls_{k}": v for k, v in cm.items()})
    met.update({f"reg_{k}": v for k, v in rm.items()})

    raw = dict(logits=logits, probs=probs, preds=preds, labels=labels,
               time_preds=t_p, time_targets=t_t, meta=all_meta)
    return met, raw


# ═══════════════════════════════════════════════════════════════════════════
# per-well / per-condition breakdown
# ═══════════════════════════════════════════════════════════════════════════
def per_well_breakdown(raw, cnames):
    meta   = raw["meta"]
    labels = raw["labels"]; preds = raw["preds"]
    probs  = raw["probs"];  tp    = raw["time_preds"]; tt = raw["time_targets"]

    wi = defaultdict(list)
    for i, m in enumerate(meta):
        wi[m.get("well", "?")].append(i)
    pw = {}
    for w in sorted(wi):
        ix = np.array(wi[w])
        acc = float((labels[ix] == preds[ix]).mean())
        cm  = multiclass_metrics(probs[ix], labels[ix], cnames)
        rm  = _reg_metrics(tp[ix], tt[ix])
        cond = meta[wi[w][0]].get("condition", "?")
        pw[w] = dict(condition=cond, n=len(ix), accuracy=acc,
                     f1_macro=cm["f1_macro"],
                     precision_macro=cm["precision_macro"],
                     recall_macro=cm["recall_macro"],
                     reg_mae=rm["mae"], reg_rmse=rm["rmse"], reg_r2=rm["r2"])

    ci = defaultdict(list)
    for i, m in enumerate(meta):
        ci[m.get("condition", "?")].append(i)
    pc = {}
    for c in sorted(ci):
        ix = np.array(ci[c])
        acc = float((labels[ix] == preds[ix]).mean())
        hrs = np.array([meta[j].get("hours", 0) for j in ix])
        bins = {}
        for s in range(0, 48, 6):
            e = s + 6
            mk = (hrs >= s) & (hrs < e)
            if mk.sum():
                bins[f"{s}-{e}h"] = dict(
                    accuracy=float((labels[ix[mk]] == preds[ix[mk]]).mean()),
                    reg_mae=float(np.abs(tp[ix[mk]] - tt[ix[mk]]).mean()),
                    n=int(mk.sum()))
        cm = multiclass_metrics(probs[ix], labels[ix], cnames)
        rm = _reg_metrics(tp[ix], tt[ix])
        pc[c] = dict(n=len(ix), accuracy=acc,
                     f1_macro=cm["f1_macro"],
                     precision_macro=cm["precision_macro"],
                     recall_macro=cm["recall_macro"],
                     reg_mae=rm["mae"], reg_rmse=rm["rmse"], reg_r2=rm["r2"],
                     time_bins=bins)

    return dict(per_well=pw, per_condition=pc)


# ═══════════════════════════════════════════════════════════════════════════
# save helpers
# ═══════════════════════════════════════════════════════════════════════════
def save_per_sample(raw, cnames, path):
    recs = []
    for i in range(len(raw["labels"])):
        r = dict(
            index=i,
            true_label=int(raw["labels"][i]),
            true_class=cnames[int(raw["labels"][i])],
            pred_label=int(raw["preds"][i]),
            pred_class=cnames[int(raw["preds"][i])],
            correct=bool(raw["labels"][i] == raw["preds"][i]),
            time_pred=float(raw["time_preds"][i]),
            time_target=float(raw["time_targets"][i]),
            time_error=float(raw["time_preds"][i] - raw["time_targets"][i]),
        )
        for ci, cn in enumerate(cnames):
            r[f"prob_{cn}"] = float(raw["probs"][i, ci])
        m = raw["meta"][i] if i < len(raw["meta"]) else {}
        r["well"]        = m.get("well", "")
        r["row"]         = m.get("row", "")
        r["position"]    = m.get("position", "")
        r["condition"]   = m.get("condition", "")
        r["frame_index"] = m.get("frame_index", -1)
        r["hours"]       = m.get("hours", -1)
        recs.append(r)
    with open(path, "w") as f:
        json.dump(recs, f, indent=1)


# ═══════════════════════════════════════════════════════════════════════════
# per-epoch analysis plots
# ═══════════════════════════════════════════════════════════════════════════
def plot_per_well_accuracy(raw, cnames, title, edir):
    """Bar chart: accuracy per well, colored by condition, sorted worst→best."""
    meta = raw["meta"]; labels = raw["labels"]; preds = raw["preds"]
    wi = defaultdict(list)
    for i, m in enumerate(meta):
        wi[m.get("well", "?")].append(i)

    wells, accs, conds = [], [], []
    for w in sorted(wi):
        ix = np.array(wi[w])
        acc = float((labels[ix] == preds[ix]).mean()) * 100
        cond = meta[wi[w][0]].get("condition", "?")
        wells.append(w); accs.append(acc); conds.append(cond)

    # Sort by accuracy
    order = np.argsort(accs)
    wells = [wells[i] for i in order]
    accs = [accs[i] for i in order]
    conds = [conds[i] for i in order]

    cond_colors = {}
    cm_tab = plt.cm.Set2(np.linspace(0, 0.8, len(cnames)))
    for ci, cn in enumerate(cnames):
        cond_colors[cn.lower()] = cm_tab[ci]
        # Also map short names
        for cond_name in ["moi5", "moi1", "moi01", "mock", "infected"]:
            if cn.lower().startswith(cond_name[:3]):
                cond_colors[cond_name] = cm_tab[ci]

    fig, ax = plt.subplots(figsize=(max(8, len(wells) * 0.5), 5))
    colors = [cond_colors.get(c.lower(), "gray") for c in conds]
    bars = ax.bar(range(len(wells)), accs, color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(wells)))
    ax.set_xticklabels([f"{w}\n({c})" for w, c in zip(wells, conds)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.axhline(y=np.mean(accs), color="red", ls="--", lw=1, label=f"Mean={np.mean(accs):.1f}%")
    # Annotate bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{acc:.0f}", ha="center", va="bottom", fontsize=7)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(edir / "per_well_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_time_binned_accuracy(raw, cnames, title, edir):
    """Line chart: accuracy in 6h time bins, one line per condition."""
    meta = raw["meta"]; labels = raw["labels"]; preds = raw["preds"]
    hours = np.array([m.get("hours", 0) for m in meta])

    cond_map = defaultdict(list)
    for i, m in enumerate(meta):
        cond_map[m.get("condition", "?")].append(i)

    bins = list(range(0, 49, 6))
    bin_labels = [f"{s}-{s+6}h" for s in bins[:-1]]

    fig, ax = plt.subplots(figsize=(10, 5))
    cm_colors = plt.cm.Set1(np.linspace(0, 0.8, len(cond_map)))

    for ci, (cond, indices) in enumerate(sorted(cond_map.items())):
        ix = np.array(indices)
        bin_accs = []
        for s in bins[:-1]:
            mask = (hours[ix] >= s) & (hours[ix] < s + 6)
            if mask.sum() > 0:
                bin_accs.append(float((labels[ix[mask]] == preds[ix[mask]]).mean()) * 100)
            else:
                bin_accs.append(np.nan)
        ax.plot(range(len(bin_labels)), bin_accs, "o-", color=cm_colors[ci],
                label=cond, lw=2, ms=5)

    # Also plot overall
    overall_accs = []
    for s in bins[:-1]:
        mask = (hours >= s) & (hours < s + 6)
        if mask.sum() > 0:
            overall_accs.append(float((labels[mask] == preds[mask]).mean()) * 100)
        else:
            overall_accs.append(np.nan)
    ax.plot(range(len(bin_labels)), overall_accs, "k--", lw=2, ms=5, label="Overall")

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Time Post-Infection")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(edir / "time_binned_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_well_condition_heatmap(raw, cnames, title, edir):
    """Plate-layout heatmap: accuracy per well arranged as plate grid.
    Handles both simple wells (c1) and run-prefixed wells (Run2_c1)."""
    meta = raw["meta"]; labels = raw["labels"]; preds = raw["preds"]
    wi = defaultdict(list)
    for i, m in enumerate(meta):
        wi[m.get("well", "?")].append(i)

    # Parse well names: "Run2_c1" → (run="Run2", row="c", col=1)
    # or "c1" → (run="", row="c", col=1)
    parsed = {}
    for w in wi:
        if "_" in w and not w.startswith("_"):
            run_tag, rc = w.rsplit("_", 1)
            parsed[w] = (run_tag, rc[0], int(rc[1:]))
        else:
            parsed[w] = ("", w[0], int(w[1:]))

    runs_order = sorted(set(p[0] for p in parsed.values()))
    plate_rows_order = sorted(set(p[1] for p in parsed.values()))
    cols_order = sorted(set(p[2] for p in parsed.values()))

    # Each run gets its own row-block in the heatmap
    row_labels = []
    for run in runs_order:
        for pr in plate_rows_order:
            lbl = f"{run} {pr.upper()}" if run else f"Row {pr.upper()}"
            row_labels.append(lbl)

    nrows = len(runs_order) * len(plate_rows_order)
    ncols = len(cols_order)
    grid = np.full((nrows, ncols), np.nan)
    annot = [['' for _ in range(ncols)] for _ in range(nrows)]

    for w, indices in wi.items():
        run, pr, col = parsed[w]
        ri = runs_order.index(run) * len(plate_rows_order) + plate_rows_order.index(pr)
        ci = cols_order.index(col)
        ix = np.array(indices)
        acc = float((labels[ix] == preds[ix]).mean()) * 100
        cond = meta[indices[0]].get("condition", "?")
        grid[ri, ci] = acc
        short = f"{pr}{col}" if not run else f"{run}_{pr}{col}"
        annot[ri][ci] = f"{short}\n{cond}\n{acc:.1f}%"

    fig, ax = plt.subplots(figsize=(max(6, ncols*2), max(3, nrows*1.2)))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, label="Accuracy (%)")

    for ri in range(nrows):
        for ci in range(ncols):
            ax.text(ci, ri, annot[ri][ci], ha="center", va="center",
                    fontsize=8, fontweight="bold")

    ax.set_xticks(range(ncols))
    ax.set_xticklabels([f"Col {c}" for c in cols_order])
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(edir / "well_plate_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_regression_scatter(raw, title, edir):
    """Scatter plot: predicted vs actual time, colored by correct/wrong."""
    tp = raw["time_preds"]; tt = raw["time_targets"]
    correct = raw["labels"] == raw["preds"]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(tt[correct], tp[correct], s=1, alpha=0.15, c="green", label="Correct cls")
    ax.scatter(tt[~correct], tp[~correct], s=1, alpha=0.15, c="red", label="Wrong cls")
    ax.plot([0, 48], [0, 48], "k--", lw=1, label="Perfect")
    ax.set_xlabel("Actual Hours"); ax.set_ylabel("Predicted Hours")
    ax.set_title(title)
    ax.legend(fontsize=9, markerscale=5)
    ax.set_xlim(-1, 50); ax.set_ylim(-1, 50)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(edir / "regression_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_well_regression(raw, title, edir):
    """Bar chart: regression MAE per well, sorted worst→best."""
    meta = raw["meta"]; tp = raw["time_preds"]; tt = raw["time_targets"]
    wi = defaultdict(list)
    for i, m in enumerate(meta):
        wi[m.get("well", "?")].append(i)

    wells, maes, conds = [], [], []
    for w in sorted(wi):
        ix = np.array(wi[w])
        mae = float(np.abs(tp[ix] - tt[ix]).mean())
        cond = meta[wi[w][0]].get("condition", "?")
        wells.append(w); maes.append(mae); conds.append(cond)

    order = np.argsort(maes)[::-1]  # worst first
    wells = [wells[i] for i in order]
    maes = [maes[i] for i in order]
    conds = [conds[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(8, len(wells)*0.5), 5))
    bars = ax.bar(range(len(wells)), maes, color="steelblue", edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(wells)))
    ax.set_xticklabels([f"{w}\n({c})" for w, c in zip(wells, conds)],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAE (hours)")
    ax.set_title(title)
    ax.axhline(y=np.mean(maes), color="red", ls="--", lw=1, label=f"Mean={np.mean(maes):.1f}h")
    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{mae:.1f}", ha="center", va="bottom", fontsize=7)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(edir / "per_well_regression_mae.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(labels, preds, cnames, title, path):
    cm = confusion_matrix(labels, preds, labels=list(range(len(cnames))))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title, fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ticks = range(len(cnames))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(cnames, rotation=45, ha="right")
    ax.set_yticklabels(cnames)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    th = cm.max() / 2
    for i in range(len(cnames)):
        for j in range(len(cnames)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > th else "black", fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_train_curves(hist, d):
    ep = [h["epoch"] for h in hist]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(ep, [h["total_loss"] for h in hist], "b-")
    axs[0].set(xlabel="Epoch", ylabel="Total Loss", title="Total Loss")
    axs[0].grid(True, alpha=.3)
    axs[1].plot(ep, [h["cls_loss"]   for h in hist], "r-")
    axs[1].set(xlabel="Epoch", ylabel="Cls Loss",   title="Classification Loss")
    axs[1].grid(True, alpha=.3)
    axs[2].plot(ep, [h["reg_loss"]   for h in hist], "g-")
    axs[2].set(xlabel="Epoch", ylabel="Reg Loss",   title="Regression Loss")
    axs[2].grid(True, alpha=.3)
    plt.tight_layout()
    fig.savefig(d / "training_curves.png", dpi=150, bbox_inches="tight")
    fig.savefig(d / "training_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ep, [h["train_acc"]*100 for h in hist], "b-")
    ax.set(xlabel="Epoch", ylabel="Train Acc (%)", title="Training Accuracy")
    ax.set_ylim(0, 100); ax.grid(True, alpha=.3)
    fig2.savefig(d / "training_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


def plot_eval_curves(eh, d, cnames=None):
    if not eh:
        return
    col = {"test_internal": "#E53935", "test_external": "#1E88E5"}
    lab = {"test_internal": "Internal (held-out 30%)",
           "test_external": "External (other dataset)"}

    panels = [
        ("cls_accuracy",        "4cls Accuracy (%)",       True),
        ("cls_f1_macro",        "4cls F1 Macro",           False),
        ("cls_f1_weighted",     "4cls F1 Weighted",        False),
        ("bin_accuracy",        "Binary Accuracy (%)",     True),
        ("bin_f1_macro",        "Binary F1 Macro",         False),
        ("bin_auc",             "Binary AUC",              False),
        ("reg_mae",             "Regression MAE (h)",      False),
        ("reg_rmse",            "Regression RMSE (h)",     False),
        ("reg_r2",              "Regression R\u00b2",      False),
    ]
    fig, axs = plt.subplots(3, 3, figsize=(18, 13))
    for ax, (mk, yl, s100) in zip(axs.flat, panels):
        for tn in eh:
            if not eh[tn]:
                continue
            xs = [r["epoch"] for r in eh[tn]]
            ys = [r["metrics"].get(mk, 0) for r in eh[tn]]
            if s100:
                ys = [y * 100 for y in ys]
            ax.plot(xs, ys, "o-", color=col.get(tn, "gray"),
                    label=lab.get(tn, tn), lw=1.5, ms=4)
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl); ax.set_title(yl)
        ax.legend(fontsize=8); ax.grid(True, alpha=.3)
    plt.suptitle("Evaluation Metrics Over Training", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(d / "eval_over_epochs.png", dpi=150, bbox_inches="tight")
    fig.savefig(d / "eval_over_epochs.pdf", bbox_inches="tight")
    plt.close(fig)

    if cnames:
        cls_colors = plt.cm.Set1(np.linspace(0, 1, len(cnames)))
        for tn in eh:
            if not eh[tn]:
                continue
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            xs = [r["epoch"] for r in eh[tn]]
            for ci, cn in enumerate(cnames):
                key = f"cls_{cn}_f1"
                ys = [r["metrics"].get(key, 0) for r in eh[tn]]
                ax2.plot(xs, ys, "o-", color=cls_colors[ci],
                         label=cn, lw=1.5, ms=4)
            ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1")
            ax2.set_title(f"Per-class F1 — {lab.get(tn, tn)}")
            ax2.legend(); ax2.grid(True, alpha=.3)
            plt.tight_layout()
            fig2.savefig(d / f"per_class_f1_{tn}.png", dpi=150,
                         bbox_inches="tight")
            plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════════
# evaluate + save one test-set at one epoch
# ═══════════════════════════════════════════════════════════════════════════
def eval_and_save(model, loader, cls_crit, reg_crit, dev, clamp,
                  cnames, cw, rw, tname, epoch, odir, log):
    met, raw = evaluate(model, loader, cls_crit, reg_crit, dev, clamp,
                        cnames, cw, rw, desc=f"{tname}_E{epoch}")
    edir = odir / tname / f"epoch_{epoch:03d}"
    edir.mkdir(parents=True, exist_ok=True)

    save_per_sample(raw, cnames, edir / "per_sample_results.json")
    np.savez(edir / "predictions.npz",
             logits=raw["logits"], probs=raw["probs"], preds=raw["preds"],
             labels=raw["labels"], time_preds=raw["time_preds"],
             time_targets=raw["time_targets"])
    with open(edir / "metrics.json", "w") as f:
        json.dump(_jsonable(met), f, indent=2)

    bd = per_well_breakdown(raw, cnames)
    with open(edir / "per_well_metrics.json", "w") as f:
        json.dump(_jsonable(bd), f, indent=2)

    plot_confusion(raw["labels"], raw["preds"], cnames,
                   f"{tname} Confusion (Epoch {epoch})",
                   edir / "confusion_matrix.png")

    plot_per_well_accuracy(raw, cnames,
                           f"{tname} Per-Well Accuracy (Epoch {epoch})", edir)
    plot_time_binned_accuracy(raw, cnames,
                              f"{tname} Time-Binned Accuracy (Epoch {epoch})", edir)
    plot_well_condition_heatmap(raw, cnames,
                                f"{tname} Plate Heatmap (Epoch {epoch})", edir)
    plot_regression_scatter(raw,
                            f"{tname} Time Regression (Epoch {epoch})", edir)
    plot_per_well_regression(raw,
                             f"{tname} Per-Well MAE (Epoch {epoch})", edir)

    rpt = classification_report(raw["labels"], raw["preds"],
                                target_names=cnames, digits=4, zero_division=0)
    (edir / "classification_report.txt").write_text(rpt)

    # ── Binary (Infected vs Mock) derived from 4-class predictions ──
    if len(cnames) > 2:
        bm = compute_binary_from_4cls(raw, num_classes=len(cnames))
        bin_met = {k: v for k, v in bm.items()
                   if not isinstance(v, np.ndarray)}
        met.update(bin_met)
        with open(edir / "binary_metrics.json", "w") as f:
            json.dump(_jsonable(bin_met), f, indent=2)
        bin_cnames = ["Infected", "Mock"]
        plot_confusion(bm["bin_labels"], bm["bin_preds"], bin_cnames,
                       f"{tname} Binary Confusion (Epoch {epoch})",
                       edir / "binary_confusion_matrix.png")
        bin_rpt = classification_report(
            bm["bin_labels"], bm["bin_preds"],
            target_names=bin_cnames, digits=4, zero_division=0)
        (edir / "binary_classification_report.txt").write_text(bin_rpt)
        log.info(f"  {tname} E{epoch:03d} BINARY | "
                 f"Acc={bm['bin_accuracy']*100:.1f}%  "
                 f"F1m={bm['bin_f1_macro']:.4f}  "
                 f"AUC={bm['bin_auc']:.4f}")

    log.info(f"  {tname} E{epoch:03d} | "
             f"Acc={met.get('cls_accuracy',0)*100:.1f}% "
             f"F1m={met.get('cls_f1_macro',0):.4f} "
             f"F1w={met.get('cls_f1_weighted',0):.4f} "
             f"Pm={met.get('cls_precision_macro',0):.4f} "
             f"Rm={met.get('cls_recall_macro',0):.4f} "
             f"AUC={met.get('cls_auc_macro',0):.4f} | "
             f"MAE={met.get('reg_mae',0):.2f}h "
             f"RMSE={met.get('reg_rmse',0):.2f}h "
             f"R2={met.get('reg_r2',0):.3f}")
    if cnames:
        for cn in cnames:
            p = met.get(f'cls_{cn}_precision', 0)
            r = met.get(f'cls_{cn}_recall', 0)
            f = met.get(f'cls_{cn}_f1', 0)
            s = met.get(f'cls_{cn}_support', 0)
            log.info(f"    {cn:>8s}: P={p:.3f}  R={r:.3f}  F1={f:.3f}  n={s}")
    log.info("    Per-well:")
    for w, wm in bd["per_well"].items():
        log.info(f"      {w} ({wm['condition']:>5s}): n={wm['n']:>5d}  "
                 f"acc={wm['accuracy']*100:5.1f}%  f1={wm['f1_macro']:.3f}  "
                 f"MAE={wm['reg_mae']:.2f}h")
    log.info("    Per-condition:")
    for c, cd in bd["per_condition"].items():
        log.info(f"      {c:>5s}: n={cd['n']:>5d}  "
                 f"acc={cd['accuracy']*100:5.1f}%  f1={cd['f1_macro']:.3f}  "
                 f"MAE={cd['reg_mae']:.2f}h  R2={cd['reg_r2']:.3f}")
        for tb, tv in cd.get("time_bins", {}).items():
            log.info(f"        {tb}: acc={tv['accuracy']*100:5.1f}%  "
                     f"MAE={tv['reg_mae']:.2f}h  n={tv['n']}")
    return met


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Path to config YAML (valrun2_train or valrun3_train)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp = cfg.get("experiment_name", "val_cross")
    odir = Path("outputs") / exp / ts
    odir.mkdir(parents=True, exist_ok=True)

    log = get_logger(str(odir / "train.log"))
    log.info(f"Output: {odir}")
    log.info(f"Config:\n{json.dumps(cfg, indent=2, default=str)}")

    set_seed(cfg.get("seed", 42))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {dev}")

    # ── data ──────────────────────────────────────────────────────────────
    dcfg = cfg["data"]
    cnames = dcfg.get("class_names", ["MOI5", "MOI1", "MOI0.1", "Mock"])
    tx = build_transforms(dcfg.get("transforms", {}))

    log.info("Building datasets ...")
    if "datasets" in dcfg and dcfg.get("datasets"):
        # Row-based split with per-directory plate layouts
        log.info("  Row-split mode enabled")
        for i, ds_entry in enumerate(dcfg["datasets"], start=1):
            log.info(f"  Dataset {i}: {ds_entry['dir']}")
        log.info(f"  Train rows: {dcfg.get('train_rows', ['a','b'])}")
        log.info(f"  Test  rows: {dcfg.get('test_rows', ['c'])}")
        dsets = build_row_split_dataset(dcfg, tx)
    elif "train_dirs" in dcfg and dcfg.get("train_dirs"):
        log.info("  Multi-train mode enabled")
        for i, td in enumerate(dcfg["train_dirs"], start=1):
            log.info(f"  Train dir {i}:  {td}")
        log.info(f"  External dir: {dcfg['external_dir']}")
        dsets = build_multi_train_external_dataset(dcfg, tx)
    else:
        log.info(f"  Train dir:    {dcfg['train_dir']}")
        log.info(f"  External dir: {dcfg['external_dir']}")
        dsets = build_cross_dataset(dcfg, tx)

    test_names = [k for k in dsets if k.startswith("test_")]
    for dn in ["train"] + test_names:
        ds = dsets[dn]
        la = np.array([s.label for s in ds.samples])
        dist = {cnames[i]: int((la == i).sum()) for i in range(len(cnames))}
        wells = sorted({s.well for s in ds.samples})
        log.info(f"  {dn:16s}: {len(ds):>7,} samples  wells={wells}  {dist}")

    bs = dcfg.get("batch_size", 64)
    nw = dcfg.get("num_workers", 4)
    ebs = bs * dcfg.get("eval_batch_size_multiplier", 2)

    train_dl = DataLoader(dsets["train"], batch_size=bs, shuffle=True,
                          num_workers=nw, pin_memory=True)
    # Non-shuffled train loader for evaluation (per-well metrics)
    train_eval_dl = DataLoader(dsets["train"], batch_size=ebs, shuffle=False,
                               num_workers=nw, pin_memory=True)
    test_dls = {tn: DataLoader(dsets[tn], batch_size=ebs, shuffle=False,
                               num_workers=nw, pin_memory=True)
                for tn in test_names}

    # ── model ─────────────────────────────────────────────────────────────
    mcfg = cfg.get("model", {})
    model = build_multitask_model(mcfg).to(dev)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ngpu = torch.cuda.device_count()
    log.info(f"Model: {mcfg.get('name','resnet50')} | params={npar:,} | GPUs={ngpu}")
    if ngpu > 1:
        model = nn.DataParallel(model)
        log.info(f"  Wrapped with DataParallel on {ngpu} GPUs")

    mt = cfg.get("multitask", {})
    clamp = tuple(mt.get("clamp_range", [0, 47.5]))
    cw = float(mt.get("classification_weight", 1))
    rw = float(mt.get("regression_weight", 1))

    ocfg = cfg.get("optimizer", {})
    optim = torch.optim.AdamW(model.parameters(),
                              lr=ocfg.get("lr", 1e-4),
                              weight_decay=ocfg.get("weight_decay", 1e-4))

    tcfg = cfg.get("training", {})
    epochs = tcfg.get("epochs", 100)
    ev_int = tcfg.get("eval_interval", 5)
    scfg = cfg.get("scheduler", {})
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=scfg.get("t_max", epochs), eta_min=scfg.get("eta_min", 1e-6))

    cls_crit = nn.CrossEntropyLoss()
    reg_crit = nn.SmoothL1Loss()
    amp_on = tcfg.get("amp", True)
    gclip  = tcfg.get("grad_clip", None)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on and dev.type == "cuda")

    ckdir = odir / "checkpoints"; ckdir.mkdir(exist_ok=True)

    # ── train ─────────────────────────────────────────────────────────────
    log.info(f"\n{'='*70}")
    log.info(f"TRAINING {epochs} epochs | eval every {ev_int}")
    log.info(f"Test sets: {test_names}")
    log.info(f"{'='*70}\n")

    hist = []
    eh: Dict[str, list] = {k: [] for k in test_names}
    t0 = time.time()

    for ep in range(1, epochs + 1):
        et0 = time.time()
        tm = train_one_epoch(model, train_dl, cls_crit, reg_crit,
                             optim, scaler, dev, clamp, cw, rw,
                             amp_on, gclip, ep)
        sched.step()
        et = time.time() - et0
        lr = optim.param_groups[0]["lr"]
        rec = dict(epoch=ep, lr=lr, epoch_time_s=round(et, 1))
        rec.update(tm); hist.append(rec)

        log.info(f"E{ep:03d}/{epochs} | "
                 f"loss={tm['total_loss']:.4f} "
                 f"(cls={tm['cls_loss']:.4f} reg={tm['reg_loss']:.4f}) | "
                 f"acc={tm['train_acc']*100:.1f}% | "
                 f"lr={lr:.2e} | {et:.1f}s")

        do_eval = (ep % ev_int == 0) or (ep == epochs)
        if do_eval:
            log.info(f"\n{'-'*50}")
            log.info(f"Checkpoint + Evaluation @ epoch {ep}")
            log.info(f"{'-'*50}")

            raw_model = model.module if hasattr(model, "module") else model
            torch.save(dict(model_state=raw_model.state_dict(),
                            optimizer_state=optim.state_dict(),
                            config=cfg, epoch=ep),
                       ckdir / f"epoch_{ep:03d}.pt")
            log.info(f"  Saved checkpoint epoch_{ep:03d}.pt")

            for tn in test_names:
                m = eval_and_save(model, test_dls[tn], cls_crit, reg_crit,
                                  dev, clamp, cnames, cw, rw,
                                  tn, ep, odir, log)
                eh[tn].append(dict(epoch=ep, metrics=m))

            # Train set per-well metrics (lightweight: metrics only, no plots)
            train_met, train_raw = evaluate(
                model, train_eval_dl, cls_crit, reg_crit, dev, clamp,
                cnames, cw, rw, desc=f"train_E{ep}")
            tedir = odir / "train_eval" / f"epoch_{ep:03d}"
            tedir.mkdir(parents=True, exist_ok=True)
            with open(tedir / "metrics.json", "w") as f:
                json.dump(_jsonable(train_met), f, indent=2)
            tbd = per_well_breakdown(train_raw, cnames)
            with open(tedir / "per_well_metrics.json", "w") as f:
                json.dump(_jsonable(tbd), f, indent=2)
            log.info(f"  train   E{ep:03d} | "
                     f"Acc={train_met.get('cls_accuracy',0)*100:.1f}% "
                     f"F1m={train_met.get('cls_f1_macro',0):.4f}")
            log.info("")

    wall = time.time() - t0
    log.info(f"\nDone in {wall:.0f}s ({wall/60:.1f} min)")

    # ── save artefacts ────────────────────────────────────────────────────
    with open(odir / "training_history.json", "w") as f:
        json.dump(hist, f, indent=2)
    plot_train_curves(hist, odir)

    eh_s = {}
    for tn, recs in eh.items():
        eh_s[tn] = [dict(epoch=r["epoch"], metrics=_jsonable(r["metrics"]))
                    for r in recs]
    with open(odir / "eval_history.json", "w") as f:
        json.dump(eh_s, f, indent=2)
    plot_eval_curves(eh, odir, cnames)
    log.info("Curves + eval-over-epochs saved")

    # ── summary ───────────────────────────────────────────────────────────
    summary = dict(experiment=exp, timestamp=ts, device=str(dev),
                   epochs=epochs, eval_interval=ev_int,
                   total_time_s=round(wall, 1), params=npar,
                   sizes={k: len(v) for k, v in dsets.items()})
    for tn in test_names:
        if eh[tn]:
            lm = eh[tn][-1]["metrics"]
            summary[f"{tn}_acc"]       = lm.get("cls_accuracy", 0)
            summary[f"{tn}_f1_macro"]  = lm.get("cls_f1_macro", 0)
            summary[f"{tn}_f1_wt"]     = lm.get("cls_f1_weighted", 0)
            summary[f"{tn}_prec"]      = lm.get("cls_precision_macro", 0)
            summary[f"{tn}_rec"]       = lm.get("cls_recall_macro", 0)
            summary[f"{tn}_auc"]       = lm.get("cls_auc_macro", 0)
            summary[f"{tn}_mae"]       = lm.get("reg_mae", 0)
            summary[f"{tn}_rmse"]      = lm.get("reg_rmse", 0)
            summary[f"{tn}_r2"]        = lm.get("reg_r2", 0)
    with open(odir / "summary.json", "w") as f:
        json.dump(_jsonable(summary), f, indent=2)

    log.info(f"\n{'='*70}")
    log.info("FINAL RESULTS (last eval epoch)")
    log.info(f"{'='*70}")
    for tn in test_names:
        a   = summary.get(f"{tn}_acc", 0)
        f1  = summary.get(f"{tn}_f1_macro", 0)
        f1w = summary.get(f"{tn}_f1_wt", 0)
        p   = summary.get(f"{tn}_prec", 0)
        r   = summary.get(f"{tn}_rec", 0)
        auc = summary.get(f"{tn}_auc", 0)
        mae = summary.get(f"{tn}_mae", 0)
        rms = summary.get(f"{tn}_rmse", 0)
        r2  = summary.get(f"{tn}_r2", 0)
        log.info(f"  {tn:16s}  Acc={a*100:.1f}%  F1m={f1:.4f}  F1w={f1w:.4f}  "
                 f"P={p:.4f}  R={r:.4f}  AUC={auc:.4f} | "
                 f"MAE={mae:.2f}h  RMSE={rms:.2f}h  R2={r2:.3f}")
    log.info(f"  Output: {odir}")
    log.info("Done!")


if __name__ == "__main__":
    main()
