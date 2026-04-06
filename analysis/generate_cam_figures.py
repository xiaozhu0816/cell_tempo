"""
Generate Grad-CAM attention maps for representative test samples.

Shows 4 examples in a 2×2 layout:
  Left column:  MOI 5 (infected) — mid-phase (t≈20h) and late-phase (t≈40h)
  Right column: Mock (uninfected) — same time points

For each example: temporal pseudo-RGB input | Grad-CAM overlay

Evaluates quality before saving. Only saves if maps are non-trivial.

Output: figS8_cam_attention.pdf / .png
"""

from __future__ import annotations
import sys
import json
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
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_ROOT  = Path("/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo")
CKPT_PATH  = Path(
    "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_multiTask/outputs"
    "/rowsplit_4cls_temporal_v2/20260327-155528/checkpoints/epoch_030.pt"
)
RUN3_DIR   = Path(
    "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/DATA/GMU_cell_1023/HBMVEC"
    "/Validation_Run3_3-13-26"
)
OUT_DIR    = Path(
    "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo/paper/supplementary"
)

sys.path.insert(0, str(CODE_ROOT))
from models.multitask_resnet import MultiTaskResNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["MOI 5", "MOI 1", "MOI 0.1", "Mock"]

# ---------------------------------------------------------------------------
# Representative samples (from per_sample_results.json)
# well, position, frame_index, true_label
# ---------------------------------------------------------------------------
SAMPLES = [
    # Same spatial position (p10), contrasting conditions and time points.
    # All four strongly and correctly predicted (prob ≈ 1.0).
    # Layout: 2×2 — rows: MOI 5 / Mock; columns: mid-phase / late-phase
    {"label": "MOI 5\n(mid-phase, t=20h)",  "tiff": RUN3_DIR / "20260311hbmvecp510xvalidationrun3_s5_c1_p10_t0-95_ec.tiff", "frame": 40, "true_cls": 0},
    {"label": "MOI 5\n(late-phase, t=40h)", "tiff": RUN3_DIR / "20260311hbmvecp510xvalidationrun3_s5_c1_p10_t0-95_ec.tiff", "frame": 80, "true_cls": 0},
    {"label": "Mock\n(mid-phase, t=20h)",   "tiff": RUN3_DIR / "20260311hbmvecp510xvalidationrun3_s5_c4_p10_t0-95_ec.tiff", "frame": 40, "true_cls": 3},
    {"label": "Mock\n(late-phase, t=40h)",  "tiff": RUN3_DIR / "20260311hbmvecp510xvalidationrun3_s5_c4_p10_t0-95_ec.tiff", "frame": 80, "true_cls": 3},
]

FRAMES_PER_HOUR = 2.0
TEMPORAL_OFFSETS_HOURS = [-6, -3, 0]
CROP_FRAC = 0.05   # 5% center crop from each side
IMG_SIZE  = 512

# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_tiff_frame(tiff_path: Path, frame_idx: int) -> np.ndarray:
    with tifffile.TiffFile(str(tiff_path)) as tif:
        frame = tif.asarray(key=frame_idx).astype(np.float32)
    return frame


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 255] uint8."""
    frame = frame - frame.min()
    mx = frame.max()
    if mx > 0:
        frame /= mx
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8)


def center_crop(frame: np.ndarray, frac: float = 0.05) -> np.ndarray:
    h, w = frame.shape[:2]
    dy, dx = int(round(h * frac)), int(round(w * frac))
    return frame[dy:h-dy, dx:w-dx]


def build_pseudo_rgb(tiff_path: Path, frame_idx: int, total_frames: int = 95) -> np.ndarray:
    """Build temporal pseudo-RGB: [H, W, 3] uint8.
    Each channel is independently normalized to [0,255] so per-frame contrast
    is preserved and temporal differences appear as color variation.
    """
    offsets = [int(o * FRAMES_PER_HOUR) for o in TEMPORAL_OFFSETS_HOURS]
    channels = []
    for off in offsets:
        fi = max(0, min(total_frames - 1, frame_idx + off))
        raw = load_tiff_frame(tiff_path, fi)
        raw = center_crop(raw, CROP_FRAC)
        channels.append(normalize_frame(raw))
    stacked = np.stack(channels, axis=-1)   # [H, W, 3]
    return stacked


def enhance_for_display(rgb: np.ndarray, gamma: float = 0.7) -> np.ndarray:
    """Enhance pseudo-RGB for visual display.
    Applies per-channel percentile stretch + gamma to reveal subtle color variation.
    """
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
    """Convert [H,W,3] uint8 → normalized tensor [1,3,H,W].
    Uses training normalization: mean=0.5, std=0.25 (per checkpoint config).
    """
    pil = Image.fromarray(rgb, mode="RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.25, 0.25, 0.25]),
    ])(pil)
    return t.unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path) -> MultiTaskResNet:
    ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
    cfg  = ckpt["config"]["model"]
    model = MultiTaskResNet(
        backbone    = cfg.get("name", "resnet50"),
        pretrained  = False,
        num_classes = cfg.get("num_classes", 4),
        dropout     = cfg.get("dropout", 0.2),
        hidden_dim  = cfg.get("hidden_dim", 256),
        use_cls_conditioning = cfg.get("use_cls_conditioning", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Grad-CAM w.r.t. last conv layer (backbone.layer4[-1])."""

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

        # Global average pooling of gradients → weights
        weights = self._grad.mean(dim=(2, 3), keepdim=True)    # [1, C, 1, 1]
        cam = F.relu((weights * self._act).sum(dim=1, keepdim=True))  # [1, 1, H', W']
        cam = cam.squeeze().cpu().numpy()

        # Resize to input size
        from scipy.ndimage import zoom as ndimage_zoom
        if cam.shape != (x.shape[2], x.shape[3]):
            zy = x.shape[2] / cam.shape[0]
            zx = x.shape[3] / cam.shape[1]
            cam = ndimage_zoom(cam, (zy, zx), order=1)

        # Normalize to [0, 1]
        mn, mx = cam.min(), cam.max()
        if mx > mn:
            cam = (cam - mn) / (mx - mn)
        return cam.astype(np.float32)


def cam_overlay(rgb_hwc: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend Jet colormap CAM onto the RGB image."""
    img = rgb_hwc.astype(np.float32) / 255.0
    heat = plt.get_cmap("jet")(cam)[:, :, :3]
    blended = (1 - alpha) * img + alpha * heat
    return np.clip(blended, 0, 1)


def cam_quality_check(cam: np.ndarray, top_pct: float = 0.1) -> bool:
    """
    Return True if CAM is non-trivial:
    - Not uniformly flat
    - Top 10% of activations cover a spatially coherent region
    """
    if cam.std() < 0.05:
        return False   # essentially uniform
    threshold = np.percentile(cam, 90)
    top_mask = (cam >= threshold).astype(float)
    # Check that high-activation region is not all in one corner
    # (rough coherence: no single quadrant > 80%)
    h, w = cam.shape
    quads = [
        top_mask[:h//2, :w//2].mean(),
        top_mask[:h//2, w//2:].mean(),
        top_mask[h//2:, :w//2].mean(),
        top_mask[h//2:, w//2:].mean(),
    ]
    if max(quads) > 0.80:
        return False  # concentrated in one corner, likely artifact
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    print("Loading model...")
    model = load_model(CKPT_PATH)
    cam_fn = GradCAM(model)

    results = []
    for s in SAMPLES:
        print(f"\nProcessing: {s['label'].replace(chr(10),' ')} ...")
        tiff_path = s["tiff"]

        # Build pseudo-RGB
        rgb_hwc = build_pseudo_rgb(tiff_path, s["frame"])   # [H, W, 3]
        tensor  = to_tensor(rgb_hwc)                         # [1, 3, 512, 512]

        # Forward pass (verify correct prediction)
        with torch.no_grad():
            logits, _ = model(tensor)
        pred_cls = logits.argmax(dim=1).item()
        print(f"  True={CLASS_NAMES[s['true_cls']]}, Pred={CLASS_NAMES[pred_cls]}, "
              f"{'✓' if pred_cls == s['true_cls'] else '✗'}")

        # Grad-CAM w.r.t. true class
        tensor_grad = tensor.clone().requires_grad_(False)
        tensor_grad = tensor.detach().requires_grad_(True)
        cam = cam_fn(tensor_grad, s["true_cls"])

        quality_ok = cam_quality_check(cam)
        print(f"  CAM std={cam.std():.3f}, quality={'OK' if quality_ok else 'POOR'}")

        # Resize rgb_hwc to IMG_SIZE for overlay
        rgb_small = np.array(
            Image.fromarray(rgb_hwc, mode="RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        )
        overlay = cam_overlay(rgb_small, cam)

        # Enhanced display version (per-channel stretch + gamma)
        rgb_display = enhance_for_display(rgb_small)

        results.append({
            "label":       s["label"],
            "true_cls":    s["true_cls"],
            "pred_cls":    pred_cls,
            "rgb":         rgb_display,   # enhanced for display
            "cam":         cam,
            "overlay":     cam_overlay(rgb_display, cam),  # overlay on enhanced
            "quality_ok":  quality_ok,
        })

    # Check overall quality
    n_ok = sum(r["quality_ok"] for r in results)
    print(f"\n{n_ok}/{len(results)} CAMs pass quality check.")
    if n_ok < 2:
        print("Too few high-quality CAMs — not saving figure.")
        return False

    # ---------------------------------------------------------------------------
    # Plot: 4 rows × 2 columns (input | CAM overlay)
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(7, 12))

    col_titles = ["Temporal pseudo-RGB input", "Grad-CAM attention"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, fontweight="bold", pad=8)

    for i, r in enumerate(results):
        # Row label
        axes[i, 0].set_ylabel(r["label"], fontsize=9, rotation=0,
                               ha="right", va="center", labelpad=55)

        # Input image
        axes[i, 0].imshow(r["rgb"])
        axes[i, 0].axis("off")

        # CAM overlay
        axes[i, 1].imshow(r["overlay"])
        # Overlay text: Pred class
        status = "✓" if r["pred_cls"] == r["true_cls"] else "✗"
        axes[i, 1].set_xlabel(f"Pred: {CLASS_NAMES[r['pred_cls']]} {status}",
                               fontsize=8, labelpad=2)
        axes[i, 1].axis("off")

        # Quality flag
        if not r["quality_ok"]:
            axes[i, 1].text(0.98, 0.02, "low-quality", transform=axes[i, 1].transAxes,
                            fontsize=7, ha="right", va="bottom", color="red", alpha=0.7)

    # Add colorbar for CAM
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, 1], fraction=0.03, pad=0.03)
    cbar.set_label("Grad-CAM activation", fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["Low", "Mid", "High"])

    fig.suptitle(
        "Grad-CAM attention maps — temporal multi-task model\n"
        "Red channels in input: t−6h; Green: t−3h; Blue: t",
        fontsize=10, y=1.01
    )
    fig.tight_layout(h_pad=1.5)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"figS8_cam_attention.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved: {out}")
    plt.close()
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
