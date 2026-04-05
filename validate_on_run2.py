#!/usr/bin/env python3
"""
External validation: apply the trained multitask model to HBMVEC Run2 data.

Run2 experiment (2026-01-21):
  - Cell type: HBMVEC passage 3 (original training used passage 5)
  - 10X, 30 min interval (same as training)
  - 93 frames per TIFF → 0 – 46 hours
  - 12 wells (a1-a4, b1-b4, c1-c4): mock + 3 MOIs, 3 replicates each
  - 36 positions per well → 432 TIFFs total

Usage:
    python validate_on_run2.py \
        --model-dir  outputs/multitask_resnet50_crop5pct/20260114-170730_5fold \
        --data-dir   "path/to/HBMVEC_10X_Run2" \
        [--plate-layout mock=a1,b1,c1  moi1=a2,b2,c2  moi3=a3,b3,c3  moi5=a4,b4,c4]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constants (must match training)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 512
CROP_BORDER = 0.05
FRAMES_PER_HOUR = 2.0
MEAN = [0.5, 0.5, 0.5]
STD  = [0.25, 0.25, 0.25]
INFECTION_ONSET_HOUR = 1.0     # infected frames start at frame 2 (1 h)

# Default plate layout  (from experimental protocol)
# Column 1 = MOI 5, Column 2 = MOI 1, Column 3 = MOI 0.1, Column 4 = Mock (PBS)
# Rows a/b/c are biological replicates.
DEFAULT_LAYOUT = {
    "moi5":  ["a1", "b1", "c1"],
    "moi1":  ["a2", "b2", "c2"],
    "moi01": ["a3", "b3", "c3"],
    "mock":  ["a4", "b4", "c4"],
}

COLORS = {
    "mock":  "#2196F3",
    "moi01": "#4CAF50",
    "moi1":  "#FF9800",
    "moi5":  "#E53935",
}

# ---------------------------------------------------------------------------
# Path helpers (Linux ↔ Windows UNC)
# ---------------------------------------------------------------------------
def _linux_to_win(p: str) -> str:
    """Convert Linux isilon path to Windows UNC path."""
    if p.startswith("/isilon/datalake/gurcan_rsch/"):
        return r"\\medctr.ad.wfubmc.edu\dfs\gurcan_rsch$" + p[len("/isilon/datalake/gurcan_rsch"):].replace("/", "\\")
    return p

def _win_to_linux(p: str) -> str:
    """Convert Windows UNC path to Linux isilon path."""
    prefix = r"\\medctr.ad.wfubmc.edu\dfs\gurcan_rsch$"
    if p.startswith(prefix):
        return "/isilon/datalake/gurcan_rsch" + p[len(prefix):].replace("\\", "/")
    return p

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_model(ckpt_path: Path, device: torch.device):
    """Load a MultiTaskResNet from a checkpoint."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from models.multitask_resnet import MultiTaskResNet

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state" in state:
        sd = state["model_state"]
    elif "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state

    model = MultiTaskResNet(
        backbone="resnet50",
        pretrained=False,
        num_classes=2,
        dropout=0.2,
        hidden_dim=256,
        use_cls_conditioning=False,
    )
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model

# ---------------------------------------------------------------------------
# Image preprocessing  (must replicate training pipeline exactly)
# ---------------------------------------------------------------------------
_eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def _load_and_preprocess(tiff_path: str, frame_idx: int) -> torch.Tensor:
    """Read a single frame from a TIFF stack, crop borders, convert to tensor."""
    with tifffile.TiffFile(tiff_path) as tif:
        n_frames = len(tif.pages)
        idx = max(0, min(n_frames - 1, frame_idx))
        frame = tif.asarray(key=idx).astype(np.float32)
    return _preprocess_frame(frame)


def _preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """Crop borders, normalise, and convert a single frame array to tensor."""
    # Centre crop to remove 5% border overlap
    if CROP_BORDER > 0:
        h, w = frame.shape[:2]
        ch = int(h * CROP_BORDER)
        cw = int(w * CROP_BORDER)
        frame = frame[ch : h - ch, cw : w - cw]

    # Normalise to [0, 1]
    fmin, fmax = frame.min(), frame.max()
    if fmax > fmin:
        frame = (frame - fmin) / (fmax - fmin)
    else:
        frame = np.zeros_like(frame)

    # to uint8 RGB
    frame_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    if frame_u8.ndim == 2:
        frame_u8 = np.stack([frame_u8] * 3, axis=-1)

    img = Image.fromarray(frame_u8)
    return _eval_transform(img).unsqueeze(0)  # [1, 3, H, W]


def _load_all_frames(tiff_path: str, indices: List[int]) -> List[torch.Tensor]:
    """Read multiple frames from a TIFF file in a single open. Much faster over network."""
    tensors = []
    with tifffile.TiffFile(tiff_path) as tif:
        n_frames = len(tif.pages)
        for fi in indices:
            idx = max(0, min(n_frames - 1, fi))
            frame = tif.asarray(key=idx).astype(np.float32)
            tensors.append(_preprocess_frame(frame))
    return tensors


# ---------------------------------------------------------------------------
# Scan Run2 data
# ---------------------------------------------------------------------------
_WELL_PATTERN = re.compile(r"_s5_([a-z])(\d+)_p(\d+)_t(\d+)-(\d+)_", re.IGNORECASE)


def scan_run2(data_dir: Path) -> Dict[str, List[Path]]:
    """Return {well_id: [sorted list of TIFF paths]}."""
    well_files: Dict[str, List[Path]] = defaultdict(list)
    for p in sorted(data_dir.glob("*.tif*")):
        m = _WELL_PATTERN.search(p.name)
        if m:
            well = f"{m.group(1).lower()}{m.group(2)}"
            well_files[well].append(p)
    return dict(well_files)


# ---------------------------------------------------------------------------
# Inference on a single TIFF file (all frames)
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_tiff(model, tiff_path: str, device: torch.device,
                 frame_range: Tuple[int, int] = None,
                 stride: int = 1) -> dict:
    """
    Run the model on selected frames of a TIFF stack.

    Returns dict with arrays:
        frame_indices, hours, cls_probs (P(infected)), time_preds
    """
    with tifffile.TiffFile(tiff_path) as tif:
        n_frames = len(tif.pages)

    if frame_range is None:
        start, end = 0, n_frames - 1
    else:
        start, end = frame_range
    indices = list(range(start, end + 1, stride))

    cls_probs_list = []
    time_preds_list = []

    for fi in indices:
        x = _load_and_preprocess(tiff_path, fi).to(device)
        cls_logits, time_pred = model(x)
        prob_infected = F.softmax(cls_logits, dim=1)[0, 1].item()
        cls_probs_list.append(prob_infected)
        time_preds_list.append(time_pred.item())

    return {
        "frame_indices": np.array(indices),
        "hours": np.array(indices, dtype=np.float64) / FRAMES_PER_HOUR,
        "cls_probs": np.array(cls_probs_list),
        "time_preds": np.array(time_preds_list),
    }


@torch.no_grad()
def predict_from_tensors(model, tensors: List[torch.Tensor],
                         device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model on pre-loaded tensors. Returns (cls_probs, time_preds)."""
    cls_probs_list = []
    time_preds_list = []
    for x in tensors:
        x = x.to(device)
        cls_logits, time_pred = model(x)
        prob_infected = F.softmax(cls_logits, dim=1)[0, 1].item()
        cls_probs_list.append(prob_infected)
        time_preds_list.append(time_pred.item())
    return np.array(cls_probs_list), np.array(time_preds_list)


# ---------------------------------------------------------------------------
# Ensemble across 5 folds  (pre-load models once for speed)
# ---------------------------------------------------------------------------
def load_fold_models(model_dir: Path, device: torch.device):
    """Pre-load all fold checkpoints. Returns list of models."""
    models = []
    for fold_idx in range(1, 6):
        ckpt = model_dir / f"fold_{fold_idx}" / "checkpoints" / "best.pt"
        if not ckpt.exists():
            continue
        models.append(_load_model(ckpt, device))
        print(f"    Loaded fold {fold_idx} model")
    if not models:
        raise RuntimeError("No fold checkpoints found")
    return models


def ensemble_predict_tiff(models: list, tiff_path: str, device: torch.device,
                          frame_range=None, stride: int = 1) -> dict:
    """Average predictions from pre-loaded fold models.
    
    Reads the TIFF file ONCE, then runs all fold models on cached tensors.
    This is ~5x faster than reading the TIFF separately for each fold.
    """
    # Determine frame indices
    with tifffile.TiffFile(tiff_path) as tif:
        n_frames = len(tif.pages)
    if frame_range is None:
        start, end = 0, n_frames - 1
    else:
        start, end = frame_range
    indices = list(range(start, end + 1, stride))

    # Read all frames ONCE  (the expensive part over network)
    tensors = _load_all_frames(tiff_path, indices)

    # Run each fold model on the cached tensors (cheap — just forward passes)
    all_cls, all_time = [], []
    for model in models:
        cls_probs, time_preds = predict_from_tensors(model, tensors, device)
        all_cls.append(cls_probs)
        all_time.append(time_preds)

    return {
        "frame_indices": np.array(indices),
        "hours": np.array(indices, dtype=np.float64) / FRAMES_PER_HOUR,
        "cls_probs": np.mean(all_cls, axis=0),
        "cls_probs_std": np.std(all_cls, axis=0),
        "time_preds": np.mean(all_time, axis=0),
        "time_preds_std": np.std(all_time, axis=0),
        "n_folds": len(models),
    }


# ---------------------------------------------------------------------------
# Aggregate across positions within a well
# ---------------------------------------------------------------------------
def aggregate_well(models: list, tiff_paths: List[Path], device: torch.device,
                   stride: int = 2, max_positions: int = None) -> dict:
    """
    Run ensemble prediction on multiple positions and aggregate.
    stride=2 means every other frame → ~1-hour resolution.
    """
    paths = list(tiff_paths)
    if max_positions and len(paths) > max_positions:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(paths), max_positions, replace=False)
        paths = [paths[i] for i in sorted(idx)]

    all_cls = []
    all_time = []
    hours = None

    for i, p in enumerate(paths):
        res = ensemble_predict_tiff(models, str(p), device, stride=stride)
        all_cls.append(res["cls_probs"])
        all_time.append(res["time_preds"])
        if hours is None:
            hours = res["hours"]
        if (i + 1) % 5 == 0 or i == len(paths) - 1:
            print(f"      position {i + 1}/{len(paths)} done", flush=True)

    cls_arr = np.stack(all_cls)   # [n_positions, n_frames]
    time_arr = np.stack(all_time)

    return {
        "hours": hours,
        "cls_prob_mean": cls_arr.mean(axis=0),
        "cls_prob_std": cls_arr.std(axis=0),
        "cls_prob_all": cls_arr,
        "time_pred_mean": time_arr.mean(axis=0),
        "time_pred_std": time_arr.std(axis=0),
        "time_pred_all": time_arr,
        "n_positions": len(paths),
        "n_folds": res["n_folds"],
    }


# ---------------------------------------------------------------------------
# Parse plate layout from command-line
# ---------------------------------------------------------------------------
def parse_plate_layout(args_layout: Optional[List[str]]) -> Dict[str, List[str]]:
    """
    Parse e.g.  ['mock=a1,b1,c1', 'moi1=a2,b2,c2', ...]
    Returns dict[condition_name] -> [well_ids]
    """
    if not args_layout:
        return DEFAULT_LAYOUT

    layout = {}
    for item in args_layout:
        cond, wells_str = item.split("=")
        layout[cond.strip()] = [w.strip().lower() for w in wells_str.split(",")]
    return layout


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_temporal_curves(results: Dict[str, dict], output_dir: Path):
    """
    Plot classification probability and time prediction over hours
    for each condition (aggregated over wells/positions).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel A: P(infected) over time ---
    ax = axes[0]
    for cond, res in results.items():
        color = COLORS.get(cond, "#999")
        h = res["hours"]
        m = res["cls_prob_mean"]
        s = res["cls_prob_std"]
        ax.plot(h, m, '-', linewidth=2, color=color, label=cond)
        ax.fill_between(h, m - s, m + s, alpha=0.15, color=color)

    ax.axhline(0.5, ls='--', color='gray', lw=1, alpha=0.5, label='Decision boundary')
    ax.set_xlabel("Hours post infection", fontweight="bold", fontsize=12)
    ax.set_ylabel("P(infected)", fontweight="bold", fontsize=12)
    ax.set_title("(A) Classification — P(infected) over time",
                 fontweight="bold", loc="left", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([0, 48])
    ax.grid(True, alpha=0.3)

    # --- Panel B: Predicted time over actual time ---
    ax = axes[1]
    for cond, res in results.items():
        color = COLORS.get(cond, "#999")
        h = res["hours"]
        m = res["time_pred_mean"]
        s = res["time_pred_std"]
        ax.plot(h, m, '-', linewidth=2, color=color, label=cond)
        ax.fill_between(h, m - s, m + s, alpha=0.15, color=color)

    max_h = 48
    ax.plot([0, max_h], [0, max_h], 'k--', lw=1.5, alpha=0.5, label="Perfect")
    ax.set_xlabel("Actual hours", fontweight="bold", fontsize=12)
    ax.set_ylabel("Predicted hours", fontweight="bold", fontsize=12)
    ax.set_title("(B) Regression — Predicted vs actual time",
                 fontweight="bold", loc="left", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim([0, max_h])
    ax.set_ylim([0, max_h])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / "run2_temporal_curves.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "run2_temporal_curves.png", dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close()


def plot_per_well(well_results: Dict[str, dict], layout: dict, output_dir: Path):
    """
    Grid of per-well P(infected) curves.
    """
    # Invert layout: well -> condition
    well_to_cond = {}
    for cond, wells in layout.items():
        for w in wells:
            well_to_cond[w] = cond

    all_wells = sorted(well_results.keys())
    n = len(all_wells)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1) if n > 1 else [axes]

    for i, well in enumerate(all_wells):
        ax = axes[i]
        res = well_results[well]
        cond = well_to_cond.get(well, "unknown")
        color = COLORS.get(cond, "#999")
        ax.plot(res["hours"], res["cls_prob_mean"], color=color, lw=1.5)
        ax.fill_between(res["hours"],
                        res["cls_prob_mean"] - res["cls_prob_std"],
                        res["cls_prob_mean"] + res["cls_prob_std"],
                        alpha=0.2, color=color)
        ax.axhline(0.5, ls='--', color='gray', lw=0.8, alpha=0.5)
        ax.set_title(f"{well} ({cond})", fontsize=10, fontweight="bold")
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([0, 48])
        ax.grid(True, alpha=0.2)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Hours")
        if i % ncols == 0:
            ax.set_ylabel("P(infected)")

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Run2 Validation — Per-well P(infected) curves",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = output_dir / "run2_per_well_cls.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.savefig(output_dir / "run2_per_well_cls.png", dpi=200, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close()


def plot_summary_metrics(well_results: Dict[str, dict], layout: dict, output_dir: Path):
    """
    Bar chart: mean P(infected) for early (0-6h) and late (24-46h) per condition.
    """
    well_to_cond = {}
    for cond, wells in layout.items():
        for w in wells:
            well_to_cond[w] = cond

    early_by_cond = defaultdict(list)
    late_by_cond = defaultdict(list)

    for well, res in well_results.items():
        cond = well_to_cond.get(well, "unknown")
        h = res["hours"]
        cls_mean = res["cls_prob_mean"]
        early_mask = h <= 6
        late_mask = h >= 24
        if early_mask.any():
            early_by_cond[cond].append(cls_mean[early_mask].mean())
        if late_mask.any():
            late_by_cond[cond].append(cls_mean[late_mask].mean())

    conditions = list(layout.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Early
    ax = axes[0]
    for i, cond in enumerate(conditions):
        vals = early_by_cond.get(cond, [0])
        m, s = np.mean(vals), np.std(vals)
        ax.bar(i, m, yerr=s, color=COLORS.get(cond, "#999"), capsize=6,
               edgecolor="black", lw=0.6)
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontweight="bold")
    ax.set_ylabel("Mean P(infected)", fontweight="bold")
    ax.set_title("(A) Early phase (0–6 h)", fontweight="bold", loc="left")
    ax.set_ylim([0, 1.1])
    ax.axhline(0.5, ls='--', color='gray', lw=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Late
    ax = axes[1]
    for i, cond in enumerate(conditions):
        vals = late_by_cond.get(cond, [0])
        m, s = np.mean(vals), np.std(vals)
        ax.bar(i, m, yerr=s, color=COLORS.get(cond, "#999"), capsize=6,
               edgecolor="black", lw=0.6)
        ax.text(i, m + s + 0.02, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontweight="bold")
    ax.set_ylabel("Mean P(infected)", fontweight="bold")
    ax.set_title("(B) Late phase (24–46 h)", fontweight="bold", loc="left")
    ax.set_ylim([0, 1.1])
    ax.axhline(0.5, ls='--', color='gray', lw=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Run2 Validation — Classification Summary by Condition",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = output_dir / "run2_summary_bar.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "run2_summary_bar.png", dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Validate trained multitask model on HBMVEC Run2 data")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to 5-fold CV model directory")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to HBMVEC_10X_Run2 TIFF directory")
    parser.add_argument("--plate-layout", nargs="*", default=None,
                        help="Well-to-condition mapping, e.g. mock=a1,b1,c1 moi1=a2,b2,c2 ...")
    parser.add_argument("--stride", type=int, default=2,
                        help="Frame stride for inference (2 = every other frame = 1h resolution)")
    parser.add_argument("--max-positions", type=int, default=12,
                        help="Max positions per well (subsample if more). Set 0 for all.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <model-dir>/Run2_Validation)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    layout = parse_plate_layout(args.plate_layout)
    max_pos = args.max_positions if args.max_positions > 0 else None

    output_dir = Path(args.output_dir) if args.output_dir else model_dir / "Run2_Validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXTERNAL VALIDATION: HBMVEC Run2")
    print("=" * 70)
    print(f"  Model dir  : {model_dir}")
    print(f"  Data dir   : {data_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Device     : {device}")
    print(f"  Stride     : {args.stride} (every {args.stride * 0.5:.1f}h)")
    print(f"  Max pos/well: {max_pos or 'all'}")
    print()

    # --- Plate layout ---
    print("Plate layout:")
    well_to_cond = {}
    for cond, wells in layout.items():
        well_to_cond.update({w: cond for w in wells})
        print(f"  {cond:12s}: {', '.join(wells)}")
    print()

    # --- Scan data ---
    well_files = scan_run2(data_dir)
    total_files = sum(len(v) for v in well_files.values())
    print(f"Found {len(well_files)} wells, {total_files} TIFF files")

    # Warn about unmapped wells
    mapped_wells = set(w for ws in layout.values() for w in ws)
    for w in well_files:
        if w not in mapped_wells:
            print(f"  [WARNING] Well {w} not in plate layout — will label as 'unknown'")
    print()

    # --- Pre-load all 5 fold models once ---
    print("Loading 5-fold ensemble models ...", flush=True)
    models = load_fold_models(model_dir, device)
    print(f"  {len(models)} models loaded\n", flush=True)

    # --- Run inference per well ---
    well_results = {}
    t0 = time.time()
    for well in sorted(well_files.keys()):
        cond = well_to_cond.get(well, "unknown")
        paths = well_files[well]
        print(f"  Well {well} ({cond}): {len(paths)} positions ...", flush=True)
        res = aggregate_well(models, paths, device,
                             stride=args.stride, max_positions=max_pos)
        well_results[well] = res
        elapsed = time.time() - t0
        print(f"    P(infected) range: {res['cls_prob_mean'].min():.3f} – "
              f"{res['cls_prob_mean'].max():.3f}  "
              f"[{elapsed:.0f}s elapsed]", flush=True)

    # --- Aggregate by condition ---
    cond_results = {}
    for cond, wells in layout.items():
        # Stack all position-level data across wells of same condition
        all_cls = []
        all_time = []
        hours = None
        for w in wells:
            if w in well_results:
                wr = well_results[w]
                all_cls.append(wr["cls_prob_mean"])
                all_time.append(wr["time_pred_mean"])
                hours = wr["hours"]
        if all_cls:
            cls_arr = np.stack(all_cls)
            time_arr = np.stack(all_time)
            cond_results[cond] = {
                "hours": hours,
                "cls_prob_mean": cls_arr.mean(axis=0),
                "cls_prob_std": cls_arr.std(axis=0),
                "time_pred_mean": time_arr.mean(axis=0),
                "time_pred_std": time_arr.std(axis=0),
                "n_wells": len(all_cls),
            }

    # --- Save raw results ---
    json_out = output_dir / "run2_predictions.json"
    save_data = {}
    for well, wr in well_results.items():
        save_data[well] = {
            "condition": well_to_cond.get(well, "unknown"),
            "n_positions": wr["n_positions"],
            "n_folds": wr["n_folds"],
            "hours": wr["hours"].tolist(),
            "cls_prob_mean": wr["cls_prob_mean"].tolist(),
            "cls_prob_std": wr["cls_prob_std"].tolist(),
            "time_pred_mean": wr["time_pred_mean"].tolist(),
            "time_pred_std": wr["time_pred_std"].tolist(),
        }
    with open(json_out, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved predictions: {json_out}")

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: Mean P(infected) by condition and phase")
    print("=" * 70)
    print(f"{'Condition':15s} {'Early(0-6h)':>12s} {'Mid(6-12h)':>12s} "
          f"{'Trans(12-24h)':>14s} {'Late(24-46h)':>13s}")
    print("-" * 70)
    for cond in layout:
        if cond not in cond_results:
            continue
        cr = cond_results[cond]
        h = cr["hours"]
        m = cr["cls_prob_mean"]
        phases = [
            ("early", h <= 6),
            ("mid", (h > 6) & (h <= 12)),
            ("trans", (h > 12) & (h <= 24)),
            ("late", h > 24),
        ]
        vals = []
        for _, mask in phases:
            if mask.any():
                vals.append(f"{m[mask].mean():.4f}")
            else:
                vals.append("  N/A")
        print(f"{cond:15s} {vals[0]:>12s} {vals[1]:>12s} {vals[2]:>14s} {vals[3]:>13s}")
    print()

    # --- Generate figures ---
    print("Generating figures ...")
    plot_temporal_curves(cond_results, output_dir)
    plot_per_well(well_results, layout, output_dir)
    plot_summary_metrics(well_results, layout, output_dir)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
