"""
run2_train_single_row.py  --  Train 4-class multitask model on single row (B or C).

Row A is excluded due to cell quality issues.

Usage
-----
    # Train on Row C, external test on Row B:
    python run2_train_single_row.py --config configs/run2_trainC.yaml

    # Train on Row B, external test on Row C:
    python run2_train_single_row.py --config configs/run2_trainB.yaml
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
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── local imports ────────────────────────────────────────────────────────────
from datasets.run2_dataset import build_run2_single_row_datasets
from models import build_multitask_model
from utils import (AverageMeter, multiclass_metrics,
                   build_transforms, get_logger, load_config, set_seed)


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════
def _meta_to_list(meta_batch) -> List[Dict[str, Any]]:
    if isinstance(meta_batch, list):
        return meta_batch
    keys = list(meta_batch.keys())
    n = len(meta_batch[keys[0]])
    out = []
    for i in range(n):
        d = {}
        for k in keys:
            v = meta_batch[k][i]
            d[k] = v.item() if isinstance(v, torch.Tensor) else v
        out.append(d)
    return out


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
    axs[0].plot(ep, [h["total_loss"] for h in hist], "b-"); axs[0].set(xlabel="Epoch", ylabel="Total Loss", title="Total Loss"); axs[0].grid(True, alpha=.3)
    axs[1].plot(ep, [h["cls_loss"]   for h in hist], "r-"); axs[1].set(xlabel="Epoch", ylabel="Cls Loss",   title="Classification Loss"); axs[1].grid(True, alpha=.3)
    axs[2].plot(ep, [h["reg_loss"]   for h in hist], "g-"); axs[2].set(xlabel="Epoch", ylabel="Reg Loss",   title="Regression Loss");     axs[2].grid(True, alpha=.3)
    plt.tight_layout(); fig.savefig(d/"training_curves.png", dpi=150, bbox_inches="tight"); fig.savefig(d/"training_curves.pdf", bbox_inches="tight"); plt.close(fig)
    fig2, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ep, [h["train_acc"]*100 for h in hist], "b-")
    ax.set(xlabel="Epoch", ylabel="Train Acc (%)", title="Training Accuracy"); ax.set_ylim(0, 100); ax.grid(True, alpha=.3)
    fig2.savefig(d/"training_accuracy.png", dpi=150, bbox_inches="tight"); plt.close(fig2)


def plot_eval_curves(eh, d, cnames=None, test_names=None):
    if not eh:
        return
    # Dynamic colors based on test set names
    colors = ["#E53935", "#1E88E5", "#43A047"]
    col = {tn: colors[i % len(colors)] for i, tn in enumerate(test_names or eh.keys())}

    panels = [
        ("cls_accuracy",        "Accuracy (%)",            True),
        ("cls_f1_macro",        "F1 Macro",                False),
        ("cls_f1_weighted",     "F1 Weighted",             False),
        ("cls_precision_macro", "Precision Macro",         False),
        ("cls_recall_macro",    "Recall Macro",            False),
        ("cls_auc_macro",       "AUC Macro (OVR)",         False),
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
                    label=tn, lw=1.5, ms=4)
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl); ax.set_title(yl)
        ax.legend(fontsize=7); ax.grid(True, alpha=.3)
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
            ax2.set_title(f"Per-class F1 — {tn}")
            ax2.legend(); ax2.grid(True, alpha=.3)
            plt.tight_layout()
            fig2.savefig(d / f"per_class_f1_{tn}.png", dpi=150, bbox_inches="tight")
            plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════════
# evaluate + save
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

    rpt = classification_report(raw["labels"], raw["preds"],
                                target_names=cnames, digits=4, zero_division=0)
    (edir / "classification_report.txt").write_text(rpt)

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
    ap.add_argument("--config", required=True, help="Path to config (e.g. configs/run2_trainC.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp = cfg.get("experiment_name", "run2_single_row")
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
    train_row = dcfg["train_row"].upper()
    external_row = dcfg["external_row"].upper()
    tx = build_transforms(dcfg.get("transforms", {}))
    log.info(f"Building datasets: train on Row {train_row}, external test on Row {external_row} ...")
    dsets = build_run2_single_row_datasets(dcfg, tx)

    # Map generic names to meaningful names
    test_held_name = f"test_{train_row.lower()}"      # e.g. "test_c"
    test_ext_name  = f"test_{external_row.lower()}"   # e.g. "test_b"
    test_name_map = {
        "test_held_out": test_held_name,
        "test_external": test_ext_name,
    }

    for dn, ds in dsets.items():
        la = np.array([s.label for s in ds.samples])
        dist = {cnames[i]: int((la == i).sum()) for i in range(len(cnames))}
        wells = sorted({s.well for s in ds.samples})
        rows = sorted({s.row for s in ds.samples})
        display_name = test_name_map.get(dn, dn)
        log.info(f"  {display_name:15s}: {len(ds):>6,} samples  rows={rows}  wells={wells}  {dist}")

    bs = dcfg.get("batch_size", 64)
    nw = dcfg.get("num_workers", 4)
    ebs = bs * dcfg.get("eval_batch_size_multiplier", 2)

    train_dl = DataLoader(dsets["train"], batch_size=bs, shuffle=True,
                          num_workers=nw, pin_memory=True)
    test_dls = {
        test_held_name: DataLoader(dsets["test_held_out"], batch_size=ebs, shuffle=False, num_workers=nw, pin_memory=True),
        test_ext_name:  DataLoader(dsets["test_external"], batch_size=ebs, shuffle=False, num_workers=nw, pin_memory=True),
    }
    test_names = [test_held_name, test_ext_name]

    # ── model ─────────────────────────────────────────────────────────────
    mcfg = cfg.get("model", {})
    model = build_multitask_model(mcfg).to(dev)
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {mcfg.get('name','resnet50')} | params={npar:,}")

    mt = cfg.get("multitask", {})
    clamp = tuple(mt.get("clamp_range", [0, 46]))
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
    log.info(f"Train: Row {train_row} (70%)")
    log.info(f"Test sets: {test_held_name} (Row {train_row} 30% held-out), {test_ext_name} (Row {external_row} 100% external)")
    log.info(f"{'='*70}\n")

    hist = []
    eh: Dict[str, list] = {tn: [] for tn in test_names}
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

            torch.save(dict(model_state=model.state_dict(),
                            optimizer_state=optim.state_dict(),
                            config=cfg, epoch=ep),
                       ckdir / f"epoch_{ep:03d}.pt")
            log.info(f"  Saved checkpoint epoch_{ep:03d}.pt")

            for tn in test_names:
                m = eval_and_save(model, test_dls[tn], cls_crit, reg_crit,
                                  dev, clamp, cnames, cw, rw,
                                  tn, ep, odir, log)
                eh[tn].append(dict(epoch=ep, metrics=m))
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
    plot_eval_curves(eh, odir, cnames, test_names)
    log.info("Curves + eval-over-epochs saved")

    # ── summary ───────────────────────────────────────────────────────────
    summary = dict(experiment=exp, timestamp=ts, device=str(dev),
                   train_row=train_row, external_row=external_row,
                   epochs=epochs, eval_interval=ev_int,
                   total_time_s=round(wall, 1), params=npar,
                   sizes={"train": len(dsets["train"]),
                          test_held_name: len(dsets["test_held_out"]),
                          test_ext_name: len(dsets["test_external"])})
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
        log.info(f"  {tn:10s}  Acc={a*100:.1f}%  F1m={f1:.4f}  F1w={f1w:.4f}  "
                 f"P={p:.4f}  R={r:.4f}  AUC={auc:.4f} | "
                 f"MAE={mae:.2f}h  RMSE={rms:.2f}h  R2={r2:.3f}")
    log.info(f"  Output: {odir}")
    log.info("Done!")


if __name__ == "__main__":
    main()
