"""
Generate a zoomed-in version of figS8_cam_attention for the manuscript.

The original Grad-CAM figure (paper/supplementary/figS8_cam_attention.pdf) used
the same 5% input crop that the model was trained with, which is correct for
inference but leaves individual cells too small to see at page width. This
script reuses the exact same checkpoint, samples, and Grad-CAM pipeline as
generate_cam_figures.py. Crucially, the model still receives the full (5%
cropped) frames so the predictions and CAMs are identical to the original
figure. Only the *displayed* panels and CAM overlays are center-cropped
further (to the central ~34%) so cells are visible in print.

Output: paper/figures/figS8_cam_attention_zoom.{png,pdf}
"""

from __future__ import annotations
import sys
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_ROOT = Path("/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo")
CKPT_PATH = Path(
    "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_multiTask/outputs"
    "/rowsplit_4cls_temporal_v2/20260327-155528/checkpoints/epoch_030.pt"
)
RUN3_DIR  = Path(
    "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/DATA/GMU_cell_1023/HBMVEC"
    "/Validation_Run3_3-13-26"
)
OUT_DIR   = Path(
    "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo/paper/figures"
)

sys.path.insert(0, str(CODE_ROOT))
from models.multitask_resnet import MultiTaskResNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["MOI 5", "MOI 1", "MOI 0.1", "Mock"]

# ---------------------------------------------------------------------------
# Representative samples: same layout as generate_cam_figures.py
# ---------------------------------------------------------------------------
SAMPLES = [
    {"label": "MOI 5 (mid-phase, t=20h)",  "tiff": RUN3_DIR / "20260311hbmvecp510xvalidationrun3_s5_c1_p10_t0-95_ec.tiff", "frame": 40, "true_cls": 0},
    {"label": "MOI 5 (late-phase, t=40h)", "tiff": RUN3_DIR / "20260311hbmvecp510xvalidationrun3_s5_c1_p10_t0-95_ec.tiff", "frame": 80, "true_cls": 0},
    {"label": "Mock (mid-phase, t=20h)",   "tiff": RUN3_DIR / "20260311hbmvecp510xvalidationrun3_s5_c4_p10_t0-95_ec.tiff", "frame": 40, "true_cls": 3},
    {"label": "Mock (late-phase, t=40h)",  "tiff": RUN3_DIR / "20260311hbmvecp510xvalidationrun3_s5_c4_p10_t0-95_ec.tiff", "frame": 80, "true_cls": 3},
]

FRAMES_PER_HOUR = 2.0
TEMPORAL_OFFSETS_HOURS = [-6, -3, 0]

# Model inputs use the same crop as training (5% per side); this keeps
# predictions and Grad-CAMs identical to the original figure.
INFER_CROP_FRAC = 0.05

# The *displayed* input and overlay panels are further center-cropped to this
# fraction of the 512x512 image so cells are visible at page width.
# 0.33 => keep the central ~34% of the input (~175 px square).
DISPLAY_KEEP_FRAC = 0.34

IMG_SIZE = 512


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_tiff_frame(tiff_path: Path, frame_idx: int) -> np.ndarray:
    with tifffile.TiffFile(str(tiff_path)) as tif:
        frame = tif.asarray(key=frame_idx).astype(np.float32)
    return frame


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame = frame - frame.min()
    mx = frame.max()
    if mx > 0:
        frame /= mx
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8)


def center_crop_symmetric(frame: np.ndarray, frac_per_side: float) -> np.ndarray:
    h, w = frame.shape[:2]
    dy, dx = int(round(h * frac_per_side)), int(round(w * frac_per_side))
    return frame[dy:h - dy, dx:w - dx]


def center_keep(frame: np.ndarray, keep_frac: float) -> np.ndarray:
    """Keep the central `keep_frac` fraction of each axis (e.g. 0.34 -> 34%)."""
    h, w = frame.shape[:2]
    kh, kw = int(round(h * keep_frac)), int(round(w * keep_frac))
    y0 = (h - kh) // 2
    x0 = (w - kw) // 2
    return frame[y0:y0 + kh, x0:x0 + kw]


def build_pseudo_rgb(tiff_path: Path, frame_idx: int, total_frames: int = 95) -> np.ndarray:
    """Build the model input exactly the way generate_cam_figures.py does."""
    offsets = [int(o * FRAMES_PER_HOUR) for o in TEMPORAL_OFFSETS_HOURS]
    channels = []
    for off in offsets:
        fi = max(0, min(total_frames - 1, frame_idx + off))
        raw = load_tiff_frame(tiff_path, fi)
        raw = center_crop_symmetric(raw, INFER_CROP_FRAC)
        channels.append(normalize_frame(raw))
    return np.stack(channels, axis=-1)


def enhance_for_display(rgb: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[:, :, c].astype(np.float32)
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        if p98 > p2:
            ch = np.clip((ch - p2) / (p98 - p2), 0, 1)
        else:
            ch = ch / 255.0
        out[:, :, c] = np.power(ch, gamma)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    pil = Image.fromarray(rgb, mode="RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
    ])(pil)
    return t.unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------------------------
# Model + Grad-CAM
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path) -> MultiTaskResNet:
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    cfg = ckpt["config"]["model"]
    model = MultiTaskResNet(
        backbone=cfg.get("name", "resnet50"),
        pretrained=False,
        num_classes=cfg.get("num_classes", 4),
        dropout=cfg.get("dropout", 0.2),
        hidden_dim=cfg.get("hidden_dim", 256),
        use_cls_conditioning=cfg.get("use_cls_conditioning", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return model


class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self._act: Optional[torch.Tensor] = None
        self._grad: Optional[torch.Tensor] = None
        target_layer = model.backbone.layer4[-1]
        target_layer.register_forward_hook(self._fhook)
        target_layer.register_full_backward_hook(self._bhook)

    def _fhook(self, m, inp, out):
        self._act = out.detach()

    def _bhook(self, m, gin, gout):
        self._grad = gout[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        logits, _ = self.model(x)
        logits[0, class_idx].backward()

        if self._act is None or self._grad is None:
            return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)

        weights = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._act).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()

        from scipy.ndimage import zoom as ndimage_zoom
        if cam.shape != (x.shape[2], x.shape[3]):
            zy = x.shape[2] / cam.shape[0]
            zx = x.shape[3] / cam.shape[1]
            cam = ndimage_zoom(cam, (zy, zx), order=1)

        mn, mx = cam.min(), cam.max()
        if mx > mn:
            cam = (cam - mn) / (mx - mn)
        return cam.astype(np.float32)


def cam_overlay(rgb_hwc: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    img = rgb_hwc.astype(np.float32) / 255.0
    heat = plt.get_cmap("jet")(cam)[:, :, :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print(f"INFER_CROP_FRAC={INFER_CROP_FRAC}  DISPLAY_KEEP_FRAC={DISPLAY_KEEP_FRAC}")
    print("Loading model...")
    model = load_model(CKPT_PATH)
    cam_fn = GradCAM(model)

    results = []
    for s in SAMPLES:
        print(f"\nProcessing: {s['label']} ...")
        rgb_hwc = build_pseudo_rgb(s["tiff"], s["frame"])
        tensor = to_tensor(rgb_hwc)

        with torch.no_grad():
            logits, _ = model(tensor)
        pred_cls = logits.argmax(dim=1).item()
        print(f"  True={CLASS_NAMES[s['true_cls']]}, Pred={CLASS_NAMES[pred_cls]}, "
              f"{'OK' if pred_cls == s['true_cls'] else 'X'}")

        tensor_grad = tensor.detach().requires_grad_(True)
        cam = cam_fn(tensor_grad, s["true_cls"])
        print(f"  CAM std={cam.std():.3f}")

        # Build 512x512 display image (matching CAM resolution)
        rgb_small = np.array(
            Image.fromarray(rgb_hwc, mode="RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        )
        rgb_display = enhance_for_display(rgb_small)
        overlay     = cam_overlay(rgb_display, cam)

        # For the figure, crop the center of the already-computed panels so
        # cells are visible. Prediction and CAM values are unchanged — this is
        # a pure display crop.
        rgb_zoomed     = center_keep(rgb_display, DISPLAY_KEEP_FRAC)
        overlay_zoomed = center_keep(overlay,      DISPLAY_KEEP_FRAC)

        results.append({
            "true_cls": s["true_cls"],
            "pred_cls": pred_cls,
            "rgb":      rgb_zoomed,
            "overlay":  overlay_zoomed,
        })

    # Layout: 2 rows (mid/late) x 4 cols (MOI5 input, MOI5 CAM, Mock input, Mock CAM)
    grid = [
        [results[0], results[2]],   # mid-phase:  MOI5, Mock
        [results[1], results[3]],   # late-phase: MOI5, Mock
    ]
    row_labels = ["Mid-phase\n($t$ = 20 h)", "Late-phase\n($t$ = 40 h)"]
    col_headers = ["MOI 5 — Input", "MOI 5 — Grad-CAM",
                   "Mock — Input",  "Mock — Grad-CAM"]

    fig = plt.figure(figsize=(12.5, 6.8))
    gs = mgridspec.GridSpec(
        2, 5,
        width_ratios=[1, 1, 1, 1, 0.055],
        wspace=0.05, hspace=0.12,
        figure=fig,
    )
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)])
    cax = fig.add_subplot(gs[:, 4])

    for row_i in range(2):
        moi_r, mock_r = grid[row_i]
        axes[row_i, 0].imshow(moi_r["rgb"])
        axes[row_i, 1].imshow(moi_r["overlay"])
        axes[row_i, 2].imshow(mock_r["rgb"])
        axes[row_i, 3].imshow(mock_r["overlay"])
        # Turn off ticks but keep the axis frame so set_ylabel still renders.
        for ax in axes[row_i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[row_i, 0].set_ylabel(row_labels[row_i], fontsize=11,
                                   rotation=0, ha="right", va="center",
                                   labelpad=18)

    for j, title in enumerate(col_headers):
        axes[0, j].set_title(title, fontsize=11, fontweight="bold", pad=6)

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Grad-CAM activation", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])
    cbar.ax.tick_params(labelsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"figS8_cam_attention_zoom.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved: {out}")
    plt.close()
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
