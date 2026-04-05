#!/usr/bin/env python#!/usr/bin/env python#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-

Generate Supplementary Figures for VEEV Cell Classification Paper

=================================================================""""""



All figures use REAL data:Generate Supplementary Figures for VEEV Cell Classification PaperGenerate Supplementary Figures for VEEV Cell Classification Paper

  S1 - Full morphological time course (real microscopy TIFF frames)

  S2 - Grad-CAM attention maps (real model checkpoint + real images)=================================================================

  S3 - Confusion matrices per window (real predictions)

  S4 - Statistical analysis with CI (real per-fold temporal metrics)Supplementary Figures:

  S5 - Error distribution histograms (real regression residuals)

  S6 - t-SNE of learned features (real backbone features)All figures use REAL data:1. Fig S1: Full morphological time course (placeholder grid)



Usage:  S1  Full morphological time course   – real microscopy TIFF frames2. Fig S2: Grad-CAM attention maps (placeholder if no model/images)

    python generate_supplementary_figures.py \

        --result-dir outputs/multitask_resnet50_crop5pct/20260114-170730_5fold  S2  Grad-CAM attention maps          – real model checkpoint + real images3. Fig S3: Confusion matrices per time window

"""

from __future__ import annotations  S3  Confusion matrices per window    – real predictions4. Fig S4: Statistical analysis with confidence intervals



import argparse  S4  Statistical analysis with CI     – real per-fold temporal metrics5. Fig S5: Error distribution histograms

import json

import sys  S5  Error distribution histograms    – real regression residuals6. Fig S6: t-SNE of learned features (optional)

from pathlib import Path

from typing import Dict, List, Optional, Tuple  S6  t-SNE of learned features        – real backbone features extracted on-the-fly



import numpy as npUsage:

import matplotlib

matplotlib.use("Agg")Usage    python generate_supplementary_figures.py --result-dir outputs/.../20260114-170730_5fold

import matplotlib.pyplot as plt

from scipy import stats-----"""

from scipy.ndimage import zoom as ndimage_zoom

    python generate_supplementary_figures.py \

import tifffile

import torch        --result-dir outputs/multitask_resnet50_crop5pct/20260114-170730_5foldfrom __future__ import annotations

import torch.nn as nn

import torch.nn.functional as F"""



# -- make project importable --from __future__ import annotationsimport argparse

CODE_ROOT = Path(__file__).resolve().parent

if str(CODE_ROOT) not in sys.path:import json

    sys.path.insert(0, str(CODE_ROOT))

import argparsefrom pathlib import Path

# -- style --

plt.rcParams.update({import jsonfrom typing import Dict, List, Tuple

    "font.family": "Arial",

    "font.size": 10,import sys

    "axes.labelsize": 11,

    "axes.titlesize": 11,from pathlib import Pathimport numpy as np

    "xtick.labelsize": 9,

    "ytick.labelsize": 9,from typing import Dict, List, Optional, Tupleimport matplotlib.pyplot as plt

    "legend.fontsize": 9,

    "figure.dpi": 300,from matplotlib import cm

    "savefig.dpi": 300,

    "savefig.bbox": "tight",import numpy as npfrom scipy import stats

    "axes.linewidth": 0.8,

    "xtick.major.width": 0.8,import matplotlib

    "ytick.major.width": 0.8,

})matplotlib.use("Agg")# -----------------------------------------------------------------------------



COLORS = {import matplotlib.pyplot as plt# Style

    "mock": "#2196F3",

    "infected": "#FF9800",from scipy import stats# -----------------------------------------------------------------------------

    "early_phase": "#E0E0E0",

}from scipy.ndimage import zoom as ndimage_zoomplt.rcParams.update({



# -- Linux / Windows path translation --    'font.family': 'Arial',

_LINUX_PFX = "/isilon/datalake/gurcan_rsch/"

_WIN_PFX = r"\\medctr.ad.wfubmc.edu\dfs\gurcan_rsch$" + "\\"import tifffile    'font.size': 10,



import torch    'axes.labelsize': 11,

def _to_win(p: str) -> str:

    if p.startswith(_LINUX_PFX):import torch.nn as nn    'axes.titlesize': 11,

        return _WIN_PFX + p[len(_LINUX_PFX):].replace("/", "\\")

    return pimport torch.nn.functional as F    'xtick.labelsize': 9,



    'ytick.labelsize': 9,

# ============================================================================

# Data helpers# ── make project importable ──────────────────────────────────────────────────    'legend.fontsize': 9,

# ============================================================================

CODE_ROOT = Path(__file__).resolve().parent    'figure.dpi': 300,

def _norm_cond(raw: str) -> str:

    raw = str(raw or "").strip().lower()if str(CODE_ROOT) not in sys.path:    'savefig.dpi': 300,

    if raw in {"mock", "uninfected", "control", "negative"}:

        return "mock"    sys.path.insert(0, str(CODE_ROOT))    'savefig.bbox': 'tight',

    if raw in {"infected", "veev", "positive"}:

        return "infected"    'axes.linewidth': 0.8,

    return raw

# ── style ────────────────────────────────────────────────────────────────────    'xtick.major.width': 0.8,



def load_predictions(cv_dir: Path) -> Dict[str, np.ndarray]:plt.rcParams.update({    'ytick.major.width': 0.8,

    cp, ct, tp, tt, hrs, conds = [], [], [], [], [], []

    for fi in range(1, 6):    "font.family": "Arial",})

        npz = cv_dir / f"fold_{fi}" / "test_predictions.npz"

        if not npz.exists():    "font.size": 10,

            continue

        d = np.load(npz)    "axes.labelsize": 11,COLORS = {

        cp.append(d["cls_preds"])

        ct.append(d["cls_targets"])    "axes.titlesize": 11,    'mock': '#2196F3',

        tp.append(d["time_preds"])

        tt.append(d["time_targets"])    "xtick.labelsize": 9,    'infected': '#FF9800',

        meta = cv_dir / f"fold_{fi}" / "test_metadata.jsonl"

        if meta.exists():    "ytick.labelsize": 9,    'early_phase': '#E0E0E0',

            fh, fc = [], []

            for ln in meta.read_text("utf-8").splitlines():    "legend.fontsize": 9,}

                if not ln.strip():

                    continue    "figure.dpi": 300,

                m = json.loads(ln)

                fh.append(float(m.get("hours_since_start", float("nan"))))    "savefig.dpi": 300,

                fc.append(_norm_cond(m.get("condition", "")))

            n = len(d["cls_targets"])    "savefig.bbox": "tight",# -----------------------------------------------------------------------------

            if len(fh) != n:

                fh = [float("nan")] * n    "axes.linewidth": 0.8,# Data loading helpers

                fc = [""] * n

            hrs.append(np.asarray(fh, dtype=np.float32))    "xtick.major.width": 0.8,# -----------------------------------------------------------------------------

            conds.extend(fc)

        else:    "ytick.major.width": 0.8,

            n = len(d["cls_targets"])

            hrs.append(np.full(n, np.nan, dtype=np.float32))})def _normalize_condition(raw: str) -> str:

            conds.extend([""] * n)

    if not cp:    raw = str(raw or '').strip().lower()

        empty = np.array([])

        return dict(cls_probs=empty, cls_targets=empty, time_preds=empty,COLORS = {    if raw in {'mock', 'uninfected', 'control', 'negative'}:

                    time_targets=empty, hours=empty, conditions=empty)

    ca = np.concatenate(ct)    "mock": "#2196F3",        return 'mock'

    co = np.asarray(conds, dtype=object)

    if co.size != ca.size:    "infected": "#FF9800",    if raw in {'infected', 'veev', 'positive'}:

        co = np.where(ca == 1, "infected", "mock")

    return dict(    "early_phase": "#E0E0E0",        return 'infected'

        cls_probs=np.concatenate(cp),

        cls_targets=ca.astype(int),}    return raw

        time_preds=np.concatenate(tp),

        time_targets=np.concatenate(tt),

        hours=np.concatenate(hrs) if hrs else np.full_like(ca, np.nan, dtype=np.float32),

        conditions=co,# ── Linux↔Windows path translation ──────────────────────────────────────────

    )

_LINUX_PFX = "/isilon/datalake/gurcan_rsch/"def load_predictions_with_metadata(cv_dir: Path) -> Dict[str, np.ndarray]:



def load_temporal_metrics(cv_dir: Path):_WIN_PFX   = r"\\medctr.ad.wfubmc.edu\dfs\gurcan_rsch$" + "\\"    """Load predictions and metadata from all folds with aligned ordering."""

    per_fold = {k: [] for k in ("accuracy", "f1", "precision", "recall", "auc")}

    wc: List[float] = []    cls_probs: List[np.ndarray] = []

    for fi in range(1, 6):

        mf = cv_dir / f"fold_{fi}" / "temporal_metrics.json"    cls_targets: List[np.ndarray] = []

        if not mf.exists():

            continuedef _to_win(p: str) -> str:    time_preds: List[np.ndarray] = []

        d = json.loads(mf.read_text("utf-8"))

        wc = d.get("window_centers", wc)    if p.startswith(_LINUX_PFX):    time_targets: List[np.ndarray] = []

        mm = d.get("metrics", {})

        for k in per_fold:        return _WIN_PFX + p[len(_LINUX_PFX):].replace("/", "\\")    hours: List[np.ndarray] = []

            if k in mm:

                per_fold[k].append(mm[k])    return p    conditions: List[str] = []

    return wc, per_fold





# ============================================================================    for fold_idx in range(1, 6):

# TIFF / image helpers

# ============================================================================# ═════════════════════════════════════════════════════════════════════════════        fold_dir = cv_dir / f"fold_{fold_idx}"



def _read_frame(tiff_path: str, frame_idx: int) -> np.ndarray:#  Data helpers        npz_file = fold_dir / "test_predictions.npz"

    with tifffile.TiffFile(tiff_path) as tif:

        idx = max(0, min(len(tif.pages) - 1, frame_idx))# ═════════════════════════════════════════════════════════════════════════════        if not npz_file.exists():

        frame = tif.asarray(key=idx)

    return frame.astype(np.float32)            continue



def _norm_cond(raw: str) -> str:        data = np.load(npz_file)

def _crop_border(f: np.ndarray, frac: float = 0.05) -> np.ndarray:

    if frac <= 0:    raw = str(raw or "").strip().lower()        cls_probs.append(data['cls_preds'])

        return f

    h, w = f.shape[-2], f.shape[-1]    if raw in {"mock", "uninfected", "control", "negative"}:        cls_targets.append(data['cls_targets'])

    dy, dx = int(round(h * frac)), int(round(w * frac))

    return f[..., dy:max(dy + 1, h - dy), dx:max(dx + 1, w - dx)]        return "mock"        time_preds.append(data['time_preds'])



    if raw in {"infected", "veev", "positive"}:        time_targets.append(data['time_targets'])

def _stretch(f: np.ndarray, lo_pct=1.0, hi_pct=99.0) -> np.ndarray:

    f = f.astype(np.float32)        return "infected"

    lo, hi = np.percentile(f, lo_pct), np.percentile(f, hi_pct)

    if hi <= lo:    return raw        meta_file = fold_dir / "test_metadata.jsonl"

        hi = lo + 1.0

    return np.clip((f - lo) / (hi - lo), 0, 1)        if meta_file.exists():



            fold_hours = []

def _to3(f: np.ndarray) -> np.ndarray:

    return np.stack([f] * 3, axis=-1) if f.ndim == 2 else fdef load_predictions(cv_dir: Path) -> Dict[str, np.ndarray]:            fold_conditions = []



    cp, ct, tp, tt, hrs, conds = [], [], [], [], [], []            with open(meta_file, 'r', encoding='utf-8') as f:

def _to_tensor(img3: np.ndarray, size: int = 512) -> torch.Tensor:

    from torchvision import transforms as T    for fi in range(1, 6):                for line in f:

    from PIL import Image

    u8 = (img3 * 255).astype(np.uint8)        npz = cv_dir / f"fold_{fi}" / "test_predictions.npz"                    if not line.strip():

    pil = Image.fromarray(u8)

    t = T.Compose([T.Resize((size, size)), T.ToTensor(),        if not npz.exists():                        continue

                   T.Normalize([0.5] * 3, [0.25] * 3)])

    return t(pil).unsqueeze(0)            continue                    meta = json.loads(line)



        d = np.load(npz)                    fold_hours.append(float(meta.get('hours_since_start', np.nan)))

# ============================================================================

# Grad-CAM        cp.append(d["cls_preds"]); ct.append(d["cls_targets"])                    fold_conditions.append(_normalize_condition(meta.get('condition', '')))

# ============================================================================

        tp.append(d["time_preds"]); tt.append(d["time_targets"])            n = len(data['cls_targets'])

class GradCAM:

    def __init__(self, model: nn.Module, layer: nn.Module):        meta = cv_dir / f"fold_{fi}" / "test_metadata.jsonl"            if len(fold_hours) != n:

        self.model = model

        self._a: Optional[torch.Tensor] = None        if meta.exists():                fold_hours = [np.nan] * n

        self._g: Optional[torch.Tensor] = None

        layer.register_forward_hook(self._fh)            fh, fc = [], []                fold_conditions = [''] * n

        layer.register_full_backward_hook(self._bh)

            for ln in meta.read_text("utf-8").splitlines():            hours.append(np.asarray(fold_hours, dtype=np.float32))

    def _fh(self, m, i, o):

        self._a = o.detach()                if not ln.strip():            conditions.extend(fold_conditions)



    def _bh(self, m, gi, go):                    continue        else:

        self._g = go[0].detach()

                m = json.loads(ln)            n = len(data['cls_targets'])

    def __call__(self, x: torch.Tensor) -> np.ndarray:

        if self._a is None or self._g is None:                fh.append(float(m.get("hours_since_start", float("nan"))))            hours.append(np.full(n, np.nan, dtype=np.float32))

            return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)

        w = self._g.mean(dim=(2, 3), keepdim=True)                fc.append(_norm_cond(m.get("condition", "")))            conditions.extend([''] * n)

        cam = F.relu((w * self._a).sum(1, keepdim=True)).squeeze().cpu().numpy()

        if cam.shape != (x.shape[2], x.shape[3]):            n = len(d["cls_targets"])

            cam = ndimage_zoom(cam, (x.shape[2] / cam.shape[0],

                                     x.shape[3] / cam.shape[1]), order=1)            if len(fh) != n:    if not cls_probs:

        mn, mx = cam.min(), cam.max()

        if mx > mn:                fh = [float("nan")] * n; fc = [""] * n        return {

            cam = (cam - mn) / (mx - mn)

        return cam.astype(np.float32)            hrs.append(np.asarray(fh, dtype=np.float32)); conds.extend(fc)            'cls_probs': np.array([]),



        else:            'cls_targets': np.array([]),

def _cam_overlay(img3: np.ndarray, cam: np.ndarray, alpha=0.4) -> np.ndarray:

    heat = plt.get_cmap("jet")(cam)[:, :, :3]            n = len(d["cls_targets"])            'time_preds': np.array([]),

    return np.clip((1 - alpha) * img3 + alpha * heat, 0, 1)

            hrs.append(np.full(n, np.nan, dtype=np.float32)); conds.extend([""] * n)            'time_targets': np.array([]),



# ============================================================================    if not cp:            'hours': np.array([]),

# Model loader

# ============================================================================        empty = np.array([])            'conditions': np.array([]),



def _load_model(ckpt_path: Path, device: torch.device) -> nn.Module:        return dict(cls_probs=empty, cls_targets=empty, time_preds=empty,        }

    from models.multitask_resnet import build_multitask_model

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)                    time_targets=empty, hours=empty, conditions=empty)

    cfg = ckpt.get("config", {}).get("model", ckpt.get("config", {}))

    model = build_multitask_model(cfg).to(device)    ca = np.concatenate(ct)    cls_probs_arr = np.concatenate(cls_probs)

    model.load_state_dict(ckpt["model_state"])

    model.eval()    co = np.asarray(conds, dtype=object)    cls_targets_arr = np.concatenate(cls_targets)

    return model

    if co.size != ca.size:    time_preds_arr = np.concatenate(time_preds)



# ============================================================================        co = np.where(ca == 1, "infected", "mock")    time_targets_arr = np.concatenate(time_targets)

# Pick representative test samples

# ============================================================================    return dict(    hours_arr = np.concatenate(hours) if hours else np.full_like(time_targets_arr, np.nan, dtype=np.float32)



def _pick_sample(cv_dir: Path, cond: str, target_h: float, tol: float = 3.5):        cls_probs=np.concatenate(cp), cls_targets=ca.astype(int),    conditions_arr = np.asarray(conditions, dtype=object)

    best, best_d = None, float("inf")

    for fi in range(1, 6):        time_preds=np.concatenate(tp), time_targets=np.concatenate(tt),

        mf = cv_dir / f"fold_{fi}" / "test_metadata.jsonl"

        if not mf.exists():        hours=np.concatenate(hrs) if hrs else np.full_like(ca, np.nan, dtype=np.float32),    if conditions_arr.size != cls_targets_arr.size:

            continue

        for ln in mf.read_text("utf-8").splitlines():        conditions=co,        conditions_arr = np.where(cls_targets_arr == 1, 'infected', 'mock')

            if not ln.strip():

                continue    )

            m = json.loads(ln)

            if _norm_cond(m.get("condition", "")) != cond:    return {

                continue

            h = float(m.get("hours_since_start", -1))        'cls_probs': cls_probs_arr,

            d = abs(h - target_h)

            if d < best_d and d <= tol:def load_temporal_metrics(cv_dir: Path):        'cls_targets': cls_targets_arr.astype(int),

                best_d = d

                best = m    per_fold = {k: [] for k in ("accuracy", "f1", "precision", "recall", "auc")}        'time_preds': time_preds_arr,

                best["_fold"] = fi

    return best    wc: List[float] = []        'time_targets': time_targets_arr,



    for fi in range(1, 6):        'hours': hours_arr,

def _display_frame(sample: dict) -> np.ndarray:

    wp = _to_win(sample["path"])        mf = cv_dir / f"fold_{fi}" / "temporal_metrics.json"        'conditions': conditions_arr,

    f = _read_frame(wp, sample["frame_index"])

    return _stretch(_crop_border(f))        if not mf.exists():    }



            continue

# ============================================================================

# Fig S1 - Full morphological time course (REAL images)        d = json.loads(mf.read_text("utf-8"))

# ============================================================================

        wc = d.get("window_centers", wc)def load_temporal_metrics(cv_dir: Path) -> Tuple[List[float], Dict[str, List[List[float]]]]:

def fig_s1(out: Path, cv_dir: Path):

    tps = list(range(0, 49, 6))        mm = d.get("metrics", {})    """Load per-fold temporal metrics for statistical analysis."""

    nc = len(tps)

    fig, axes = plt.subplots(2, nc, figsize=(2.2 * nc, 5))        for k in per_fold:    per_fold: Dict[str, List[List[float]]] = {

    for row, (label, cond) in enumerate([("Mock (PBS)", "mock"),

                                          ("VEEV (TC-83)", "infected")]):            if k in mm:        'accuracy': [],

        for col, t in enumerate(tps):

            ax = axes[row, col]                per_fold[k].append(mm[k])        'f1': [],

            s = _pick_sample(cv_dir, cond, float(t))

            if s is not None:    return wc, per_fold        'precision': [],

                try:

                    ax.imshow(_display_frame(s), cmap="gray", vmin=0, vmax=1)        'recall': [],

                except Exception as e:

                    ax.text(0.5, 0.5, "err", ha="center", va="center",        'auc': [],

                            fontsize=6, color="red", transform=ax.transAxes)

            else:# ═════════════════════════════════════════════════════════════════════════════    }

                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8,

                        color="gray", transform=ax.transAxes)#  TIFF / image helpers    window_centers: List[float] = []

            ax.set_xticks([])

            ax.set_yticks([])# ═════════════════════════════════════════════════════════════════════════════

            if row == 0:

                ax.set_title(f"{t} h", fontsize=9, fontweight="bold")    for fold_idx in range(1, 6):

        axes[row, 0].set_ylabel(label, fontsize=10, fontweight="bold")

    fig.suptitle("Fig S1. Full Morphological Time Course",def _read_frame(tiff_path: str, frame_idx: int) -> np.ndarray:        metrics_file = cv_dir / f"fold_{fold_idx}" / "temporal_metrics.json"

                 fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])    with tifffile.TiffFile(tiff_path) as tif:        if not metrics_file.exists():

    for ext in ("pdf", "png"):

        fig.savefig(out / f"fig_s1_morphology_full.{ext}")        idx = max(0, min(len(tif.pages) - 1, frame_idx))            continue

    plt.close(fig)

    print("  [OK] Fig S1 saved (real microscopy)")        frame = tif.asarray(key=idx)        data = json.loads(metrics_file.read_text(encoding='utf-8'))



    return frame.astype(np.float32)        window_centers = data.get('window_centers', window_centers)

# ============================================================================

# Fig S2 - Grad-CAM (REAL model + REAL images)        metrics = data.get('metrics', {})

# ============================================================================

        for key in per_fold.keys():

def fig_s2(out: Path, cv_dir: Path):

    device = torch.device("cpu")def _crop_border(f: np.ndarray, frac: float = 0.05) -> np.ndarray:            if key in metrics:

    ckpt = cv_dir / "fold_1" / "checkpoints" / "best.pt"

    if not ckpt.exists():    if frac <= 0:                per_fold[key].append(metrics[key])

        print("  [SKIP] S2 (no checkpoint)")

        return        return f



    print("  Loading model for Grad-CAM ...")    h, w = f.shape[-2], f.shape[-1]    return window_centers, per_fold

    model = _load_model(ckpt, device)

    cam_obj = GradCAM(model, model.backbone.layer4[-1])    dy, dx = int(round(h * frac)), int(round(w * frac))



    tps = [0, 12, 24, 36, 48]    return f[..., dy:max(dy + 1, h - dy), dx:max(dx + 1, w - dx)]

    nc = len(tps)

    fig, axes = plt.subplots(4, nc, figsize=(3.0 * nc, 12))# -----------------------------------------------------------------------------



    for col, t in enumerate(tps):# Figure S1: Full Morphological Time Course (placeholders)

        for pair_idx, (cond, lbl_raw, lbl_cam) in enumerate([

            ("mock", "Mock (PBS)", "Mock Grad-CAM"),def _stretch(f: np.ndarray, lo_pct=1.0, hi_pct=99.0) -> np.ndarray:# -----------------------------------------------------------------------------

            ("infected", "VEEV (TC-83)", "VEEV Grad-CAM"),

        ]):    f = f.astype(np.float32)

            r_raw = pair_idx * 2

            r_cam = pair_idx * 2 + 1    lo, hi = np.percentile(f, lo_pct), np.percentile(f, hi_pct)def fig_s1_full_timecourse(output_dir: Path, timepoints: List[int] | None = None) -> None:

            s = _pick_sample(cv_dir, cond, float(t), tol=4.0)

            if s is None:    if hi <= lo:    if timepoints is None:

                for r in (r_raw, r_cam):

                    axes[r, col].text(0.5, 0.5, "N/A", ha="center", va="center",        hi = lo + 1.0        timepoints = list(range(0, 49, 6))

                                      fontsize=8, color="gray",

                                      transform=axes[r, col].transAxes)    return np.clip((f - lo) / (hi - lo), 0, 1)

                    axes[r, col].set_xticks([])

                    axes[r, col].set_yticks([])    n_cols = len(timepoints)

                continue

            try:    fig, axes = plt.subplots(2, n_cols, figsize=(2.2 * n_cols, 5))

                f01 = _display_frame(s)

                img3 = _to3(f01)def _to3(f: np.ndarray) -> np.ndarray:

                axes[r_raw, col].imshow(f01, cmap="gray", vmin=0, vmax=1)

    return np.stack([f] * 3, axis=-1) if f.ndim == 2 else f    for row, label in enumerate(['Mock', 'VEEV']):

                tgt_cls = 1 if cond == "infected" else 0

                x = _to_tensor(img3).to(device)        for col, t in enumerate(timepoints):

                x.requires_grad_(True)

                cls_logits, _ = model(x)            ax = axes[row, col] if n_cols > 1 else axes[row]

                cls_logits[:, tgt_cls].sum().backward()

                cam_map = cam_obj(x)def _to_tensor(img3: np.ndarray, size: int = 512) -> torch.Tensor:            ax.set_xticks([])

                model.zero_grad(set_to_none=True)

    from torchvision import transforms as T            ax.set_yticks([])

                dh, dw = f01.shape[:2]

                if cam_map.shape != (dh, dw):    from PIL import Image            ax.set_xlim(0, 1)

                    cam_map = ndimage_zoom(cam_map,

                                           (dh / cam_map.shape[0],    u8 = (img3 * 255).astype(np.uint8)            ax.set_ylim(0, 1)

                                            dw / cam_map.shape[1]), order=1)

                axes[r_cam, col].imshow(_cam_overlay(img3, cam_map))    pil = Image.fromarray(u8)            ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, lw=1.0, color='gray'))



                p_inf = torch.softmax(cls_logits, 1)[0, 1].item()    t = T.Compose([T.Resize((size, size)), T.ToTensor(),            ax.text(0.5, 0.5, 'Image\nplaceholder', ha='center', va='center', fontsize=8, color='gray')

                axes[r_cam, col].text(

                    0.98, 0.02, f"P(inf)={p_inf:.2f}",                   T.Normalize([0.5]*3, [0.25]*3)])            if row == 0:

                    transform=axes[r_cam, col].transAxes, ha="right", va="bottom",

                    fontsize=7, color="white",    return t(pil).unsqueeze(0)                ax.set_title(f"{t}h", fontsize=9, fontweight='bold')

                    bbox=dict(facecolor="black", alpha=0.6, pad=2))

            except Exception as e:        axes[row, 0].set_ylabel(label, fontsize=10, fontweight='bold', rotation=90)

                print(f"    [WARN] cam error {cond} {t}h: {e}")

                for r in (r_raw, r_cam):

                    axes[r, col].text(0.5, 0.5, "err", ha="center", va="center",

                                      fontsize=7, color="red",# ═════════════════════════════════════════════════════════════════════════════    fig.suptitle('Fig S1. Full Morphological Time Course (placeholders)', fontsize=12, fontweight='bold')

                                      transform=axes[r, col].transAxes)

            for r in (r_raw, r_cam):#  Grad-CAM    plt.tight_layout(rect=[0, 0, 1, 0.95])

                axes[r, col].set_xticks([])

                axes[r, col].set_yticks([])# ═════════════════════════════════════════════════════════════════════════════    plt.savefig(output_dir / 'fig_s1_morphology_full.pdf')

        axes[0, col].set_title(f"{t} h", fontsize=11, fontweight="bold")

    plt.savefig(output_dir / 'fig_s1_morphology_full.png')

    for r, lbl in enumerate(["Mock (PBS)", "Mock Grad-CAM",

                              "VEEV (TC-83)", "VEEV Grad-CAM"]):class GradCAM:    plt.close()

        axes[r, 0].set_ylabel(lbl, fontsize=10, fontweight="bold")

    fig.suptitle("Fig S2. Grad-CAM Attention Maps", fontsize=13, fontweight="bold")    def __init__(self, model: nn.Module, layer: nn.Module):

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ("pdf", "png"):        self.model = model

        fig.savefig(out / f"fig_s2_gradcam.{ext}")

    plt.close(fig)        self._a: Optional[torch.Tensor] = None# -----------------------------------------------------------------------------

    print("  [OK] Fig S2 saved (real Grad-CAM)")

        self._g: Optional[torch.Tensor] = None# Figure S2: Grad-CAM (placeholder)



# ============================================================================        layer.register_forward_hook(self._fh)# -----------------------------------------------------------------------------

# Fig S3 - Confusion matrices per time window

# ============================================================================        layer.register_full_backward_hook(self._bh)



def fig_s3(out: Path, data: Dict[str, np.ndarray]):def _synthetic_cell_image(seed: int = 0, size: int = 224) -> np.ndarray:

    hours = data["hours"]

    y = data["cls_targets"]    def _fh(self, m, i, o):    rng = np.random.default_rng(seed)

    p = data["cls_probs"]

    if len(hours) == 0:        self._a = o.detach()    base = rng.normal(0.5, 0.1, size=(size, size))

        print("  [SKIP] S3")

        return    base = np.clip(base, 0, 1)



    stride, wsz = 3.0, 6.0    def _bh(self, m, gi, go):    for _ in range(6):

    centers = np.arange(3, 46, stride)

    nr, nc_grid = 3, 5        self._g = go[0].detach()        cx, cy = rng.integers(0, size, size=2)

    fig, axes = plt.subplots(nr, nc_grid, figsize=(16, 9))

    axes_flat = axes.flatten()        radius = rng.integers(10, 30)



    for idx, c in enumerate(centers):    def __call__(self, x: torch.Tensor) -> np.ndarray:        y, x = np.ogrid[:size, :size]

        mask = (hours >= c - wsz / 2) & (hours < c + wsz / 2)

        cm = np.zeros((2, 2), dtype=int)        if self._a is None or self._g is None:        mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

        if mask.sum() > 0:

            yw = y[mask]            return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)        base[mask] = np.clip(base[mask] + rng.uniform(0.2, 0.4), 0, 1)

            pw = (p[mask] >= 0.5).astype(int)

            for t_val, p_val in zip(yw, pw):        w = self._g.mean(dim=(2, 3), keepdim=True)    return base

                cm[int(t_val), int(p_val)] += 1

        ax = axes_flat[idx]        cam = F.relu((w * self._a).sum(1, keepdim=True)).squeeze().cpu().numpy()

        ax.imshow(cm, cmap="Blues")

        tot = max(cm.sum(), 1)        if cam.shape != (x.shape[2], x.shape[3]):

        for i in range(2):

            for j in range(2):            cam = ndimage_zoom(cam, (x.shape[2] / cam.shape[0],def _synthetic_cam(seed: int = 0, size: int = 224) -> np.ndarray:

                ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / tot * 100:.1f}%)",

                        ha="center", va="center", fontsize=8)                                     x.shape[3] / cam.shape[1]), order=1)    rng = np.random.default_rng(seed)

        ax.set_title(f"{int(c)}h", fontsize=9)

        ax.set_xticks([0, 1])        mn, mx = cam.min(), cam.max()    cam = rng.normal(0.0, 1.0, size=(size, size))

        ax.set_yticks([0, 1])

        ax.set_xticklabels(["Mock", "Inf."], fontsize=7)        return ((cam - mn) / (mx - mn)).astype(np.float32) if mx > mn else cam.astype(np.float32)    cam = np.exp(-((cam - cam.mean()) ** 2) / (2 * cam.std() ** 2))

        ax.set_yticklabels(["Mock", "Inf."], fontsize=7)

    for idx in range(len(centers), nr * nc_grid):    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        axes_flat[idx].axis("off")

    return cam

    fig.suptitle("Fig S3. Confusion Matrices by Time Window",

                 fontsize=12, fontweight="bold")def _cam_overlay(img3: np.ndarray, cam: np.ndarray, alpha=0.4) -> np.ndarray:

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("pdf", "png"):    heat = plt.get_cmap("jet")(cam)[:, :, :3]

        fig.savefig(out / f"fig_s3_confusion_matrices.{ext}")

    plt.close(fig)    return np.clip((1 - alpha) * img3 + alpha * heat, 0, 1)def fig_s2_gradcam(output_dir: Path) -> None:

    print("  [OK] Fig S3 saved")

    timepoints = ['0h', '12h', '24h', '36h']



# ============================================================================    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Fig S4 - Statistical analysis (CI + paired t-test)

# ============================================================================# ═════════════════════════════════════════════════════════════════════════════



def fig_s4(out: Path, cv_dir: Path):#  Model loader    for col, tp in enumerate(timepoints):

    centers, pf = load_temporal_metrics(cv_dir)

    if not centers or not pf["accuracy"]:# ═════════════════════════════════════════════════════════════════════════════        img = _synthetic_cell_image(seed=col)

        print("  [SKIP] S4")

        return        infected = _synthetic_cell_image(seed=10 + col)



    ca = np.asarray(centers, dtype=float)def _load_model(ckpt_path: Path, device: torch.device) -> nn.Module:        cam = _synthetic_cam(seed=20 + col)

    res: Dict[str, dict] = {}

    for metric, vals in pf.items():    from models.multitask_resnet import build_multitask_model        heatmap = plt.get_cmap('jet')(cam)[:, :, :3]

        mat = np.asarray(vals, dtype=float)

        res[metric] = {}    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)        overlay = 0.6 * np.stack([infected] * 3, axis=-1) + 0.4 * heatmap

        for i, c in enumerate(ca):

            v = mat[:, i]    cfg = ckpt.get("config", {}).get("model", ckpt.get("config", {}))

            mn = v.mean()

            sem = stats.sem(v)    model = build_multitask_model(cfg).to(device)        axes[0, col].imshow(img, cmap='gray')

            if not np.isfinite(sem) or sem == 0:

                ci = (mn, mn)    model.load_state_dict(ckpt["model_state"])        axes[1, col].imshow(infected, cmap='gray')

            else:

                ci = stats.t.interval(0.95, len(v) - 1, loc=mn, scale=sem)    model.eval()        axes[2, col].imshow(overlay)

            res[metric][c] = dict(mean=mn, ci_lo=ci[0], ci_hi=ci[1])

    return model

    # early vs late t-test

    early_m = ca <= 6        axes[0, col].set_title(tp, fontsize=10, fontweight='bold')

    late_m = ca >= 24

    ea = np.nanmean(np.asarray(pf["accuracy"])[:, early_m], axis=1)

    la = np.nanmean(np.asarray(pf["accuracy"])[:, late_m], axis=1)

    t_stat, p_val = stats.ttest_rel(ea, la)# ═════════════════════════════════════════════════════════════════════════════    for row, label in enumerate(['Mock', 'VEEV', 'Grad-CAM']):



    # LaTeX table#  Pick representative test samples        axes[row, 0].set_ylabel(label, fontsize=11, fontweight='bold')

    tbl = ["\\begin{tabular}{lccc}", "\\hline",

           "Window & Accuracy & F1 & AUC \\\\ ", "\\hline"]# ═════════════════════════════════════════════════════════════════════════════

    for c in ca:

        tbl.append(    for ax in axes.flatten():

            f"{int(c)}h & {res['accuracy'][c]['mean']:.4f}"

            f" & {res['f1'][c]['mean']:.4f}"def _pick_sample(cv_dir: Path, cond: str, target_h: float, tol: float = 3.5):        ax.set_xticks([])

            f" & {res['auc'][c]['mean']:.4f} \\\\ "

        )    best, best_d = None, float("inf")        ax.set_yticks([])

    tbl += ["\\hline", "\\end{tabular}"]

    (out / "fig_s4_temporal_stats_table.tex").write_text("\n".join(tbl), "utf-8")    for fi in range(1, 6):



    summary = (        mf = cv_dir / f"fold_{fi}" / "test_metadata.jsonl"    fig.suptitle('Fig S2. Grad-CAM Attention Maps (placeholder)', fontsize=12, fontweight='bold')

        f"Early vs Late (Accuracy): t={t_stat:.3f}, p={p_val:.4f}\n"

        f"Early mean={ea.mean():.4f}, Late mean={la.mean():.4f}\n"        if not mf.exists():    plt.tight_layout(rect=[0, 0, 1, 0.95])

    )

    (out / "fig_s4_temporal_stats_summary.txt").write_text(summary, "utf-8")            continue    plt.savefig(output_dir / 'fig_s2_gradcam.pdf')



    fig, ax = plt.subplots(figsize=(10, 6))        for ln in mf.read_text("utf-8").splitlines():    plt.savefig(output_dir / 'fig_s2_gradcam.png')

    for metric, color, marker in [

        ("accuracy", COLORS["mock"], "o"),            if not ln.strip():    plt.close()

        ("f1", "#4CAF50", "s"),

        ("auc", COLORS["infected"], "^"),                continue

    ]:

        means = [res[metric][c]["mean"] for c in ca]            m = json.loads(ln)

        ci_lo = [res[metric][c]["ci_lo"] for c in ca]

        ci_hi = [res[metric][c]["ci_hi"] for c in ca]            if _norm_cond(m.get("condition", "")) != cond:# -----------------------------------------------------------------------------

        ax.plot(ca, means, marker=marker, color=color, label=metric.upper())

        ax.fill_between(ca, ci_lo, ci_hi, color=color, alpha=0.2)                continue# Figure S3: Confusion Matrices per Time Window

    ax.axhline(0.95, color="gray", ls="--", alpha=0.6)

    ax.axhline(0.99, color="gray", ls="--", alpha=0.6)            h = float(m.get("hours_since_start", -1))# -----------------------------------------------------------------------------

    ax.axvspan(0, 6, color=COLORS["early_phase"], alpha=0.35, label="Early Phase")

    ax.set_xlabel("Time Window Center (hours)")            d = abs(h - target_h)

    ax.set_ylabel("Performance Metric")

    ax.set_ylim(0.85, 1.02)            if d < best_d and d <= tol:def fig_s3_confusion_matrices(output_dir: Path, data: Dict[str, np.ndarray]) -> None:

    ax.legend()

    ax.set_title("Fig S4. Temporal Metrics with 95% CI")                best_d = d; best = m; best["_fold"] = fi    hours = data['hours']

    plt.tight_layout()

    for ext in ("pdf", "png"):    return best    y_true = data['cls_targets']

        fig.savefig(out / f"fig_s4_statistics.{ext}")

    plt.close(fig)    p_inf = data['cls_probs']

    print("  [OK] Fig S4 saved")





# ============================================================================def _display_frame(sample: dict) -> np.ndarray:    if len(hours) == 0:

# Fig S5 - Error distribution histograms

# ============================================================================    wp = _to_win(sample["path"])        print("⚠ Skipping Fig S3 (no hours data)")



def fig_s5(out: Path, data: Dict[str, np.ndarray]):    f = _read_frame(wp, sample["frame_index"])        return

    if data["time_targets"].size == 0:

        print("  [SKIP] S5")    return _stretch(_crop_border(f))

        return

    window_size = 6.0

    err = np.abs(data["time_preds"] - data["time_targets"])

    y = data["cls_targets"]    stride = 3.0



    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))# ═════════════════════════════════════════════════════════════════════════════    centers = np.arange(3, 46, stride)

    axes[0].hist(err, bins=50, color="steelblue", alpha=0.8, edgecolor="black")

    axes[0].set_title("All Samples")#  Fig S1 – Full morphological time course (REAL images)

    axes[0].set_xlabel("Absolute Error (hours)")

    axes[0].set_ylabel("Count")# ═════════════════════════════════════════════════════════════════════════════    fig, axes = plt.subplots(3, 5, figsize=(16, 9))



    axes[1].hist(err[y == 0], bins=40, color=COLORS["mock"], alpha=0.7, label="Mock")    axes = axes.flatten()

    axes[1].hist(err[y == 1], bins=40, color=COLORS["infected"], alpha=0.7, label="Infected")

    axes[1].set_title("By Condition")def fig_s1(out: Path, cv_dir: Path):

    axes[1].set_xlabel("Absolute Error (hours)")

    axes[1].legend()    tps = list(range(0, 49, 6))    for idx, center in enumerate(centers):



    fig.suptitle("Fig S5. Error Distributions", fontsize=12, fontweight="bold")    nc = len(tps)        win_start = center - window_size / 2

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    for ext in ("pdf", "png"):    fig, axes = plt.subplots(2, nc, figsize=(2.2 * nc, 5))        win_end = center + window_size / 2

        fig.savefig(out / f"fig_s5_errors.{ext}")

    plt.close(fig)    for row, (label, cond) in enumerate([("Mock (PBS)", "mock"),        mask = (hours >= win_start) & (hours < win_end)

    print("  [OK] Fig S5 saved")

                                          ("VEEV (TC-83)", "infected")]):        if mask.sum() == 0:



# ============================================================================        for col, t in enumerate(tps):            cm = np.zeros((2, 2), dtype=int)

# Fig S6 - t-SNE of learned features (REAL extraction)

# ============================================================================            ax = axes[row, col]        else:



def fig_s6(out: Path, cv_dir: Path, max_samples: int = 2000):            s = _pick_sample(cv_dir, cond, float(t))            y_win = y_true[mask]

    """Extract backbone features on test samples and plot t-SNE."""

    device = torch.device("cpu")            if s is not None:            pred_win = (p_inf[mask] >= 0.5).astype(int)

    ckpt = cv_dir / "fold_1" / "checkpoints" / "best.pt"

    if not ckpt.exists():                try:            cm = np.zeros((2, 2), dtype=int)

        print("  [SKIP] S6 (no checkpoint)")

        return                    ax.imshow(_display_frame(s), cmap="gray", vmin=0, vmax=1)            for t, p in zip(y_win, pred_win):

    try:

        from sklearn.manifold import TSNE                except Exception as e:                cm[int(t), int(p)] += 1

    except ImportError:

        print("  [SKIP] S6 (sklearn not available)")                    ax.text(.5, .5, "err", ha="center", va="center",

        return

                            fontsize=6, color="red", transform=ax.transAxes)        ax = axes[idx]

    print("  Extracting features for t-SNE ...")

    model = _load_model(ckpt, device)            else:        ax.imshow(cm, cmap='Blues')



    meta_file = cv_dir / "fold_1" / "test_metadata.jsonl"                ax.text(.5, .5, "N/A", ha="center", va="center", fontsize=8,        total = cm.sum() if cm.sum() > 0 else 1

    if not meta_file.exists():

        print("  [SKIP] S6 (no metadata)")                        color="gray", transform=ax.transAxes)        for i in range(2):

        return

            ax.set_xticks([]); ax.set_yticks([])            for j in range(2):

    samples = []

    for ln in meta_file.read_text("utf-8").splitlines():            if row == 0:                pct = cm[i, j] / total * 100

        if not ln.strip():

            continue                ax.set_title(f"{t} h", fontsize=9, fontweight="bold")                ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha='center', va='center', fontsize=8)

        samples.append(json.loads(ln))

        axes[row, 0].set_ylabel(label, fontsize=10, fontweight="bold")        ax.set_title(f"{int(center)}h", fontsize=9)

    rng = np.random.default_rng(42)

    if len(samples) > max_samples:    fig.suptitle("Fig S1. Full Morphological Time Course",        ax.set_xticks([0, 1])

        idx = rng.choice(len(samples), max_samples, replace=False)

        samples = [samples[i] for i in idx]                 fontsize=12, fontweight="bold")        ax.set_yticks([0, 1])



    feats_list, labels_list, hours_list = [], [], []    plt.tight_layout(rect=[0, 0, 1, 0.95])        ax.set_xticklabels(['Mock', 'Inf.'], fontsize=7)

    with torch.no_grad():

        for i, s in enumerate(samples):    for ext in ("pdf", "png"):        ax.set_yticklabels(['Mock', 'Inf.'], fontsize=7)

            try:

                wp = _to_win(s["path"])        fig.savefig(out / f"fig_s1_morphology_full.{ext}")

                f = _read_frame(wp, s["frame_index"])

                f01 = _stretch(_crop_border(f))    plt.close(fig)    fig.suptitle('Fig S3. Confusion Matrices by Time Window', fontsize=12, fontweight='bold')

                x = _to_tensor(_to3(f01)).to(device)

                feat = model.get_features(x).squeeze().cpu().numpy()    print("  \u2713 Fig S1 saved (real microscopy)")    plt.tight_layout(rect=[0, 0, 1, 0.95])

                feats_list.append(feat)

                labels_list.append(    plt.savefig(output_dir / 'fig_s3_confusion_matrices.pdf')

                    1 if _norm_cond(s.get("condition", "")) == "infected" else 0

                )    plt.savefig(output_dir / 'fig_s3_confusion_matrices.png')

                hours_list.append(float(s.get("hours_since_start", 0)))

            except Exception:# ═════════════════════════════════════════════════════════════════════════════    plt.close()

                continue

            if (i + 1) % 200 == 0:#  Fig S2 – Grad-CAM (REAL model + REAL images)

                print(f"    {i + 1}/{len(samples)} ...")

# ═════════════════════════════════════════════════════════════════════════════

    if len(feats_list) < 50:

        print("  [SKIP] S6 (too few valid samples)")# -----------------------------------------------------------------------------

        return

def fig_s2(out: Path, cv_dir: Path):# Figure S4: Statistical Analysis (CI + tests)

    feats = np.stack(feats_list)

    labels = np.array(labels_list)    device = torch.device("cpu")# -----------------------------------------------------------------------------

    hours_arr = np.array(hours_list)

    ckpt = cv_dir / "fold_1" / "checkpoints" / "best.pt"

    print(f"  Running t-SNE on {feats.shape[0]} samples x {feats.shape[1]} dims ...")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)    if not ckpt.exists():def fig_s4_statistical_analysis(output_dir: Path, cv_dir: Path) -> None:

    emb = tsne.fit_transform(feats)

        print("  \u26a0 Skip S2 (no checkpoint)"); return    centers, per_fold = load_temporal_metrics(cv_dir)

    np.savez(out / "tsne_features.npz", features=feats, labels=labels,

             hours=hours_arr, embedding=emb)    if not centers or not per_fold['accuracy']:



    fig, axes = plt.subplots(1, 2, figsize=(12, 5))    print("  Loading model for Grad-CAM ...")        print("⚠ Skipping Fig S4 (missing per-fold temporal metrics)")



    # Panel A: colored by condition    model = _load_model(ckpt, device)        return

    ax = axes[0]

    ax.scatter(emb[labels == 0, 0], emb[labels == 0, 1],    cam_obj = GradCAM(model, model.backbone.layer4[-1])

               s=12, c=COLORS["mock"], label="Mock", alpha=0.6)

    ax.scatter(emb[labels == 1, 0], emb[labels == 1, 1],    centers_arr = np.asarray(centers, dtype=float)

               s=12, c=COLORS["infected"], label="Infected", alpha=0.6)

    ax.legend()    tps = [0, 12, 24, 36, 48]    results = {m: {} for m in per_fold.keys()}

    ax.set_xticks([])

    ax.set_yticks([])    nc = len(tps)

    ax.set_title("Colored by Condition")

    ax.text(-0.05, 1.02, "(A)", transform=ax.transAxes,    fig, axes = plt.subplots(4, nc, figsize=(3.0 * nc, 12))    for metric, values in per_fold.items():

            fontsize=12, fontweight="bold")

        mat = np.asarray(values, dtype=float)

    # Panel B: colored by time

    ax2 = axes[1]    for col, t in enumerate(tps):        for idx, center in enumerate(centers_arr):

    sc = ax2.scatter(emb[:, 0], emb[:, 1], s=12, c=hours_arr,

                     cmap="viridis", alpha=0.6)        for pair_idx, (cond, lbl_raw, lbl_cam) in enumerate([            vals = mat[:, idx]

    plt.colorbar(sc, ax=ax2, label="Hours")

    ax2.set_xticks([])            ("mock",     "Mock (PBS)",   "Mock Grad-CAM"),            mean = np.mean(vals)

    ax2.set_yticks([])

    ax2.set_title("Colored by Time (hours)")            ("infected", "VEEV (TC-83)", "VEEV Grad-CAM"),            sd = np.std(vals, ddof=1)

    ax2.text(-0.05, 1.02, "(B)", transform=ax2.transAxes,

             fontsize=12, fontweight="bold")        ]):            sem = stats.sem(vals)



    fig.suptitle("Fig S6. t-SNE of Learned Features",            r_raw = pair_idx * 2            if not np.isfinite(sem) or sem == 0:

                 fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])            r_cam = pair_idx * 2 + 1                ci = (mean, mean)

    for ext in ("pdf", "png"):

        fig.savefig(out / f"fig_s6_tsne.{ext}")            s = _pick_sample(cv_dir, cond, float(t), tol=4.0)            else:

    plt.close(fig)

    print("  [OK] Fig S6 saved (real features)")            if s is None:                ci = stats.t.interval(0.95, len(vals) - 1, loc=mean, scale=sem)



                for r in (r_raw, r_cam):            results[metric][center] = {

# ============================================================================

# Main                    axes[r, col].text(.5, .5, "N/A", ha="center", va="center",                'mean': mean,

# ============================================================================

                                      fontsize=8, color="gray",                'sd': sd,

def main() -> int:

    parser = argparse.ArgumentParser()                                      transform=axes[r, col].transAxes)                'ci_lower': ci[0],

    parser.add_argument("--result-dir", type=str, required=True)

    args = parser.parse_args()                    axes[r, col].set_xticks([]); axes[r, col].set_yticks([])                'ci_upper': ci[1],



    cv_dir = Path(args.result_dir)                continue                'min': float(np.min(vals)),

    out = cv_dir / "Figures" / "Supplementary"

    out.mkdir(parents=True, exist_ok=True)            try:                'max': float(np.max(vals)),



    data = load_predictions(cv_dir)                f01 = _display_frame(s)            }



    print("Generating supplementary figures (real data) ...")                img3 = _to3(f01)

    fig_s1(out, cv_dir)

    fig_s2(out, cv_dir)                axes[r_raw, col].imshow(f01, cmap="gray", vmin=0, vmax=1)    # Early vs late t-test (accuracy)

    fig_s3(out, data)

    fig_s4(out, cv_dir)    early_mask = centers_arr <= 6

    fig_s5(out, data)

    fig_s6(out, cv_dir)                tgt_cls = 1 if cond == "infected" else 0    late_mask = centers_arr >= 24



    print(f"\nAll supplementary figures saved to: {out}")                x = _to_tensor(img3).to(device); x.requires_grad_(True)    early_vals = np.nanmean(np.asarray(per_fold['accuracy'])[:, early_mask], axis=1)

    return 0

                cls_logits, _ = model(x)    late_vals = np.nanmean(np.asarray(per_fold['accuracy'])[:, late_mask], axis=1)



if __name__ == "__main__":                cls_logits[:, tgt_cls].sum().backward()    t_stat, p_val = stats.ttest_rel(early_vals, late_vals)

    raise SystemExit(main())

                cam_map = cam_obj(x)

                model.zero_grad(set_to_none=True)    # Save LaTeX table (subset for readability)

    table_lines = ["\\begin{tabular}{lccc}", "\\hline", "Window & Accuracy & F1 & AUC \\\\ ", "\\hline"]

                dh, dw = f01.shape[:2]    for c in centers_arr:

                if cam_map.shape != (dh, dw):        acc = results['accuracy'][c]['mean']

                    cam_map = ndimage_zoom(cam_map, (dh / cam_map.shape[0],        f1 = results['f1'][c]['mean']

                                                      dw / cam_map.shape[1]), order=1)        auc = results['auc'][c]['mean']

                axes[r_cam, col].imshow(_cam_overlay(img3, cam_map))    table_lines.append(f"{int(c)}h & {acc:.4f} & {f1:.4f} & {auc:.4f} \\\\ ")

    table_lines.append("\\hline")

                p_inf = torch.softmax(cls_logits, 1)[0, 1].item()    table_lines.append("\\end{tabular}")

                axes[r_cam, col].text(    (output_dir / 'fig_s4_temporal_stats_table.tex').write_text("\n".join(table_lines), encoding='utf-8')

                    .98, .02, f"P(inf)={p_inf:.2f}",

                    transform=axes[r_cam, col].transAxes, ha="right", va="bottom",    summary = (

                    fontsize=7, color="white",        f"Early vs Late (Accuracy): t={t_stat:.3f}, p={p_val:.4f}\n"

                    bbox=dict(facecolor="black", alpha=.6, pad=2))        f"Early mean={early_vals.mean():.4f}, Late mean={late_vals.mean():.4f}\n"

            except Exception as e:    )

                print(f"    \u26a0 cam error {cond} {t}h: {e}")    (output_dir / 'fig_s4_temporal_stats_summary.txt').write_text(summary, encoding='utf-8')

                for r in (r_raw, r_cam):

                    axes[r, col].text(.5, .5, "err", ha="center", va="center",    # Plot with CI

                                      fontsize=7, color="red",    fig, ax = plt.subplots(figsize=(10, 6))

                                      transform=axes[r, col].transAxes)    for metric, color, marker in [

            for r in (r_raw, r_cam):        ('accuracy', COLORS['mock'], 'o'),

                axes[r, col].set_xticks([]); axes[r, col].set_yticks([])        ('f1', '#4CAF50', 's'),

        axes[0, col].set_title(f"{t} h", fontsize=11, fontweight="bold")        ('auc', COLORS['infected'], '^'),

    ]:

    for r, lbl in enumerate(["Mock (PBS)", "Mock Grad-CAM",        means = [results[metric][c]['mean'] for c in centers_arr]

                              "VEEV (TC-83)", "VEEV Grad-CAM"]):        ci_lower = [results[metric][c]['ci_lower'] for c in centers_arr]

        axes[r, 0].set_ylabel(lbl, fontsize=10, fontweight="bold")        ci_upper = [results[metric][c]['ci_upper'] for c in centers_arr]

    fig.suptitle("Fig S2. Grad-CAM Attention Maps", fontsize=13, fontweight="bold")        ax.plot(centers_arr, means, marker=marker, color=color, label=metric.upper())

    plt.tight_layout(rect=[0, 0, 1, 0.96])        ax.fill_between(centers_arr, ci_lower, ci_upper, color=color, alpha=0.2)

    for ext in ("pdf", "png"):

        fig.savefig(out / f"fig_s2_gradcam.{ext}")    ax.axhline(0.95, color='gray', linestyle='--', alpha=0.6)

    plt.close(fig)    ax.axhline(0.99, color='gray', linestyle='--', alpha=0.6)

    print("  \u2713 Fig S2 saved (real Grad-CAM)")    ax.axvspan(0, 6, color=COLORS['early_phase'], alpha=0.35, label='Early Phase')

    ax.set_xlabel('Time Window Center (hours)')

    ax.set_ylabel('Performance Metric')

# ═════════════════════════════════════════════════════════════════════════════    ax.set_ylim(0.85, 1.02)

#  Fig S3 – Confusion matrices per time window    ax.legend()

# ═════════════════════════════════════════════════════════════════════════════    ax.set_title('Fig S4. Temporal Metrics with 95% CI')



def fig_s3(out: Path, data: Dict[str, np.ndarray]):    plt.tight_layout()

    hours = data["hours"]; y = data["cls_targets"]; p = data["cls_probs"]    plt.savefig(output_dir / 'fig_s4_statistics.pdf')

    if len(hours) == 0:    plt.savefig(output_dir / 'fig_s4_statistics.png')

        print("  \u26a0 Skip S3"); return    plt.close()



    stride, wsz = 3.0, 6.0

    centers = np.arange(3, 46, stride)# -----------------------------------------------------------------------------

    nr = 3; nc = 5# Figure S5: Error Distribution Histograms

    fig, axes = plt.subplots(nr, nc, figsize=(16, 9))# -----------------------------------------------------------------------------

    axes_flat = axes.flatten()

def fig_s5_error_distribution(output_dir: Path, data: Dict[str, np.ndarray]) -> None:

    for idx, c in enumerate(centers):    if data['time_targets'].size == 0:

        mask = (hours >= c - wsz / 2) & (hours < c + wsz / 2)        print("⚠ Skipping Fig S5 (no regression data)")

        cm = np.zeros((2, 2), dtype=int)        return

        if mask.sum() > 0:

            yw, pw = y[mask], (p[mask] >= .5).astype(int)    errors = np.abs(data['time_preds'] - data['time_targets'])

            for t, pp in zip(yw, pw):    y = data['cls_targets']

                cm[int(t), int(pp)] += 1

        ax = axes_flat[idx]    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax.imshow(cm, cmap="Blues")    axes[0].hist(errors, bins=50, color='steelblue', alpha=0.8, edgecolor='black')

        tot = max(cm.sum(), 1)    axes[0].set_title('All Samples')

        for i in range(2):    axes[0].set_xlabel('Absolute Error (hours)')

            for j in range(2):    axes[0].set_ylabel('Count')

                ax.text(j, i, f"{cm[i,j]}\n({cm[i,j]/tot*100:.1f}%)",

                        ha="center", va="center", fontsize=8)    axes[1].hist(errors[y == 0], bins=40, color=COLORS['mock'], alpha=0.7, label='Mock')

        ax.set_title(f"{int(c)}h", fontsize=9)    axes[1].hist(errors[y == 1], bins=40, color=COLORS['infected'], alpha=0.7, label='Infected')

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])    axes[1].set_title('By Condition')

        ax.set_xticklabels(["Mock", "Inf."], fontsize=7)    axes[1].set_xlabel('Absolute Error (hours)')

        ax.set_yticklabels(["Mock", "Inf."], fontsize=7)    axes[1].legend()

    # hide unused

    for idx in range(len(centers), nr * nc):    fig.suptitle('Fig S5. Error Distributions', fontsize=12, fontweight='bold')

        axes_flat[idx].axis("off")    plt.tight_layout(rect=[0, 0, 1, 0.92])

    plt.savefig(output_dir / 'fig_s5_errors.pdf')

    fig.suptitle("Fig S3. Confusion Matrices by Time Window",    plt.savefig(output_dir / 'fig_s5_errors.png')

                 fontsize=12, fontweight="bold")    plt.close()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("pdf", "png"):

        fig.savefig(out / f"fig_s3_confusion_matrices.{ext}")# -----------------------------------------------------------------------------

    plt.close(fig)# Figure S6: t-SNE (optional)

    print("  \u2713 Fig S3 saved")# -----------------------------------------------------------------------------



def fig_s6_tsne(output_dir: Path, feature_path: Path | None) -> None:

# ═════════════════════════════════════════════════════════════════════════════    if feature_path is None or not feature_path.exists():

#  Fig S4 – Statistical analysis (CI + paired t-test)        print("⚠ Skipping Fig S6 (feature file not provided)")

# ═════════════════════════════════════════════════════════════════════════════        return



def fig_s4(out: Path, cv_dir: Path):    try:

    centers, pf = load_temporal_metrics(cv_dir)        from sklearn.manifold import TSNE

    if not centers or not pf["accuracy"]:    except Exception:

        print("  \u26a0 Skip S4"); return        print("⚠ Skipping Fig S6 (scikit-learn not available)")

        return

    ca = np.asarray(centers, dtype=float)

    res: Dict[str, dict] = {}    data = np.load(feature_path)

    for metric, vals in pf.items():    feats = data['features'] if 'features' in data else None

        mat = np.asarray(vals, dtype=float)    labels = data['labels'] if 'labels' in data else None

        res[metric] = {}    if feats is None or labels is None:

        for i, c in enumerate(ca):        print("⚠ Skipping Fig S6 (features/labels missing)")

            v = mat[:, i]        return

            mn = v.mean(); sem = stats.sem(v)

            ci = (mn, mn) if (not np.isfinite(sem) or sem == 0) \    tsne = TSNE(n_components=2, perplexity=30, random_state=42)

                 else stats.t.interval(0.95, len(v) - 1, loc=mn, scale=sem)    emb = tsne.fit_transform(feats)

            res[metric][c] = dict(mean=mn, ci_lo=ci[0], ci_hi=ci[1])

    fig, ax = plt.subplots(figsize=(6, 5))

    # early vs late t-test    ax.scatter(emb[labels == 0, 0], emb[labels == 0, 1], s=10, c=COLORS['mock'], label='Mock', alpha=0.7)

    early_m = ca <= 6; late_m = ca >= 24    ax.scatter(emb[labels == 1, 0], emb[labels == 1, 1], s=10, c=COLORS['infected'], label='Infected', alpha=0.7)

    ea = np.nanmean(np.asarray(pf["accuracy"])[:, early_m], axis=1)    ax.set_title('Fig S6. t-SNE of Learned Features')

    la = np.nanmean(np.asarray(pf["accuracy"])[:, late_m], axis=1)    ax.legend()

    t_stat, p_val = stats.ttest_rel(ea, la)    ax.set_xticks([])

    ax.set_yticks([])

    # LaTeX table

    tbl = ["\\begin{tabular}{lccc}", "\\hline",    plt.tight_layout()

           "Window & Accuracy & F1 & AUC \\\\ ", "\\hline"]    plt.savefig(output_dir / 'fig_s6_tsne.pdf')

    for c in ca:    plt.savefig(output_dir / 'fig_s6_tsne.png')

        tbl.append(f"{int(c)}h & {res['accuracy'][c]['mean']:.4f}"    plt.close()

                   f" & {res['f1'][c]['mean']:.4f}"

                   f" & {res['auc'][c]['mean']:.4f} \\\\ ")

    tbl += ["\\hline", "\\end{tabular}"]# -----------------------------------------------------------------------------

    (out / "fig_s4_temporal_stats_table.tex").write_text("\n".join(tbl), "utf-8")# Main

# -----------------------------------------------------------------------------

    summary = (f"Early vs Late (Accuracy): t={t_stat:.3f}, p={p_val:.4f}\n"

               f"Early mean={ea.mean():.4f}, Late mean={la.mean():.4f}\n")def main() -> int:

    (out / "fig_s4_temporal_stats_summary.txt").write_text(summary, "utf-8")    parser = argparse.ArgumentParser()

    parser.add_argument('--result-dir', type=str, required=True, help='Path to CV results directory')

    fig, ax = plt.subplots(figsize=(10, 6))    parser.add_argument('--feature-path', type=str, default=None, help='Optional .npz file with features/labels for t-SNE')

    for metric, color, marker in [    args = parser.parse_args()

        ("accuracy", COLORS["mock"], "o"),

        ("f1", "#4CAF50", "s"),    cv_dir = Path(args.result_dir)

        ("auc", COLORS["infected"], "^"),    output_dir = cv_dir / 'Figures' / 'Supplementary'

    ]:    output_dir.mkdir(parents=True, exist_ok=True)

        means  = [res[metric][c]["mean"]  for c in ca]

        ci_lo  = [res[metric][c]["ci_lo"] for c in ca]    data = load_predictions_with_metadata(cv_dir)

        ci_hi  = [res[metric][c]["ci_hi"] for c in ca]

        ax.plot(ca, means, marker=marker, color=color, label=metric.upper())    fig_s1_full_timecourse(output_dir)

        ax.fill_between(ca, ci_lo, ci_hi, color=color, alpha=0.2)    fig_s2_gradcam(output_dir)

    ax.axhline(0.95, color="gray", ls="--", alpha=.6)    fig_s3_confusion_matrices(output_dir, data)

    ax.axhline(0.99, color="gray", ls="--", alpha=.6)    fig_s4_statistical_analysis(output_dir, cv_dir)

    ax.axvspan(0, 6, color=COLORS["early_phase"], alpha=.35, label="Early Phase")    fig_s5_error_distribution(output_dir, data)

    ax.set_xlabel("Time Window Center (hours)")    fig_s6_tsne(output_dir, Path(args.feature_path) if args.feature_path else None)

    ax.set_ylabel("Performance Metric")

    ax.set_ylim(0.85, 1.02)    print(f"\nSupplementary figures saved to: {output_dir}")

    ax.legend()    return 0

    ax.set_title("Fig S4. Temporal Metrics with 95% CI")

    plt.tight_layout()

    for ext in ("pdf", "png"):if __name__ == '__main__':

        fig.savefig(out / f"fig_s4_statistics.{ext}")    raise SystemExit(main())

    plt.close(fig)
    print("  \u2713 Fig S4 saved")


# ═════════════════════════════════════════════════════════════════════════════
#  Fig S5 – Error distribution histograms
# ═════════════════════════════════════════════════════════════════════════════

def fig_s5(out: Path, data: Dict[str, np.ndarray]):
    if data["time_targets"].size == 0:
        print("  \u26a0 Skip S5"); return

    err = np.abs(data["time_preds"] - data["time_targets"])
    y = data["cls_targets"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(err, bins=50, color="steelblue", alpha=.8, edgecolor="black")
    axes[0].set_title("All Samples"); axes[0].set_xlabel("Absolute Error (hours)")
    axes[0].set_ylabel("Count")

    axes[1].hist(err[y == 0], bins=40, color=COLORS["mock"],     alpha=.7, label="Mock")
    axes[1].hist(err[y == 1], bins=40, color=COLORS["infected"], alpha=.7, label="Infected")
    axes[1].set_title("By Condition"); axes[1].set_xlabel("Absolute Error (hours)")
    axes[1].legend()

    fig.suptitle("Fig S5. Error Distributions", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_s5_errors.{ext}")
    plt.close(fig)
    print("  \u2713 Fig S5 saved")


# ═════════════════════════════════════════════════════════════════════════════
#  Fig S6 – t-SNE of learned features (REAL extraction)
# ═════════════════════════════════════════════════════════════════════════════

def fig_s6(out: Path, cv_dir: Path, max_samples: int = 2000):
    """Extract backbone features on test samples and plot t-SNE."""
    device = torch.device("cpu")
    ckpt = cv_dir / "fold_1" / "checkpoints" / "best.pt"
    if not ckpt.exists():
        print("  \u26a0 Skip S6 (no checkpoint)"); return
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  \u26a0 Skip S6 (sklearn not available)"); return

    print("  Extracting features for t-SNE ...")
    model = _load_model(ckpt, device)

    # Collect test metadata from fold_1 for feature extraction
    meta_file = cv_dir / "fold_1" / "test_metadata.jsonl"
    if not meta_file.exists():
        print("  \u26a0 Skip S6 (no metadata)"); return

    samples = []
    for ln in meta_file.read_text("utf-8").splitlines():
        if not ln.strip():
            continue
        samples.append(json.loads(ln))

    # Subsample if too many
    rng = np.random.default_rng(42)
    if len(samples) > max_samples:
        idx = rng.choice(len(samples), max_samples, replace=False)
        samples = [samples[i] for i in idx]

    feats_list, labels_list, hours_list = [], [], []
    with torch.no_grad():
        for i, s in enumerate(samples):
            try:
                wp = _to_win(s["path"])
                f = _read_frame(wp, s["frame_index"])
                f01 = _stretch(_crop_border(f))
                x = _to_tensor(_to3(f01)).to(device)
                feat = model.get_features(x).squeeze().cpu().numpy()
                feats_list.append(feat)
                labels_list.append(1 if _norm_cond(s.get("condition", "")) == "infected" else 0)
                hours_list.append(float(s.get("hours_since_start", 0)))
            except Exception:
                continue
            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(samples)} ...")

    if len(feats_list) < 50:
        print("  \u26a0 Skip S6 (too few valid samples)"); return

    feats = np.stack(feats_list)
    labels = np.array(labels_list)
    hours_arr = np.array(hours_list)

    print(f"  Running t-SNE on {feats.shape[0]} samples x {feats.shape[1]} dims ...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    emb = tsne.fit_transform(feats)

    # Save features for reuse
    np.savez(out / "tsne_features.npz", features=feats, labels=labels,
             hours=hours_arr, embedding=emb)

    # Plot 1: colored by condition
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(emb[labels == 0, 0], emb[labels == 0, 1],
               s=12, c=COLORS["mock"], label="Mock", alpha=.6)
    ax.scatter(emb[labels == 1, 0], emb[labels == 1, 1],
               s=12, c=COLORS["infected"], label="Infected", alpha=.6)
    ax.legend(); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Colored by Condition")
    ax.text(-0.05, 1.02, "(A)", transform=ax.transAxes, fontsize=12, fontweight="bold")

    # Plot 2: colored by time
    ax2 = axes[1]
    sc = ax2.scatter(emb[:, 0], emb[:, 1], s=12, c=hours_arr,
                     cmap="viridis", alpha=.6)
    plt.colorbar(sc, ax=ax2, label="Hours")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("Colored by Time (hours)")
    ax2.text(-0.05, 1.02, "(B)", transform=ax2.transAxes, fontsize=12, fontweight="bold")

    fig.suptitle("Fig S6. t-SNE of Learned Features", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_s6_tsne.{ext}")
    plt.close(fig)
    print("  \u2713 Fig S6 saved (real features)")


# ═════════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    args = parser.parse_args()

    cv_dir = Path(args.result_dir)
    out = cv_dir / "Figures" / "Supplementary"
    out.mkdir(parents=True, exist_ok=True)

    data = load_predictions(cv_dir)

    print("Generating supplementary figures (real data) ...")
    fig_s1(out, cv_dir)
    fig_s2(out, cv_dir)
    fig_s3(out, data)
    fig_s4(out, cv_dir)
    fig_s5(out, data)
    fig_s6(out, cv_dir)

    print(f"\nAll supplementary figures saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
