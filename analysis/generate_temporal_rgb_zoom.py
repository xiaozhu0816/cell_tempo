"""
Generate a zoomed-in version of fig_temporal_rgb_examples for the manuscript.

The original figure (paper/figures/fig_temporal_rgb_examples.png) was a 3-row x
5-column grid rendered at page width, which made individual cells too small to
see. This script rebuilds the same figure but center-crops each panel to a
tight ROI and emits a wider, taller canvas so cells are clearly visible in
supplementary Figure S3.

Output: paper/figures/fig_temporal_rgb_examples_zoom.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tifffile

DATA_DIR = Path("/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/DATA/GMU_cell_1023/HBMVEC")
RUN2_DIR = DATA_DIR / "Validation_Run2_3-11-26" / "Enhanced contour"
OUT      = Path("/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo/paper/figures")

# Fraction of the image kept per side after center-crop. 0.32 = keep the
# central ~32% region, i.e. strong zoom so cells are clearly visible.
ZOOM_FRAC = 0.32

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


def find_tiff(well, pos="10"):
    for f in sorted(RUN2_DIR.glob("*.tif*")):
        if f"_{well}_" in f.name and f"_p{pos}_" in f.name:
            return f
    return None


def read_frame(tiff_path, frame_idx):
    with tifffile.TiffFile(str(tiff_path)) as tif:
        n_frames = len(tif.pages)
        fi = max(0, min(frame_idx, n_frames - 1))
        raw = tif.asarray(key=fi).astype(np.float32)
    if raw.max() > 0:
        raw = (raw - raw.min()) / (raw.max() - raw.min()) * 255
    return raw.astype(np.uint8)


def center_crop(img, frac=ZOOM_FRAC):
    h, w = img.shape[:2]
    ch, cw = int(h * frac / 2), int(w * frac / 2)
    y0, x0 = h // 2 - ch, w // 2 - cw
    return img[y0:y0 + 2 * ch, x0:x0 + 2 * cw]


def make_pseudo_rgb(tiff_path, frame_idx, offsets_hours=(-6, -3, 0), fph=2.0):
    offset_frames = [int(h * fph) for h in offsets_hours]
    channels = []
    with tifffile.TiffFile(str(tiff_path)) as tif:
        n_frames = len(tif.pages)
        for off in offset_frames:
            fi = frame_idx + off
            fi = max(0, min(fi, n_frames - 1))
            raw = tif.asarray(key=fi).astype(np.float32)
            if raw.max() > 0:
                raw = (raw - raw.min()) / (raw.max() - raw.min()) * 255
            channels.append(raw.astype(np.uint8))
    return np.stack(channels, axis=-1)


def main():
    tiff = find_tiff("a1", "10")
    if tiff is None:
        raise SystemExit("Could not locate MOI5 TIFF (a1, p10) for figure generation.")

    examples = [
        ("Early (t = 2 h)",  4,  "Channels clamped to t = 0\n(limited temporal info)"),
        ("Mid (t = 20 h)",  40,  "6 h temporal span\n(full temporal context)"),
        ("Late (t = 40 h)", 80,  "Late-stage CPE\n(strong temporal signal)"),
    ]
    offsets = (-6, -3, 0)
    fph = 2.0
    channel_labels = ["R:  t − 6 h", "G:  t − 3 h", "B:  t  (current)"]
    channel_cmaps  = ["Reds", "Greens", "Blues"]

    # Taller per-row panels so that the cropped ROI renders large.
    fig = plt.figure(figsize=(14, len(examples) * 4.0 + 0.6))
    gs = gridspec.GridSpec(
        len(examples), 5,
        width_ratios=[1, 1, 1, 0.18, 1.2],
        hspace=0.42, wspace=0.08,
    )

    with tifffile.TiffFile(str(tiff)) as tif:
        n_frames = len(tif.pages)

    for row_i, (title, frame_idx, note) in enumerate(examples):
        offset_frames = [int(h * fph) for h in offsets]
        for ch_i, (off, label, cmap) in enumerate(zip(offset_frames, channel_labels, channel_cmaps)):
            fi = frame_idx + off
            actual_fi = max(0, min(fi, n_frames - 1))
            frame = read_frame(tiff, actual_fi)
            frame = center_crop(frame)

            ax = fig.add_subplot(gs[row_i, ch_i])
            ax.imshow(frame, cmap=cmap, vmin=0, vmax=255)
            actual_h = actual_fi / fph
            clamp_note = " (clamped)" if fi != actual_fi else ""
            ax.set_title(f"{label}\nframe {actual_fi} ({actual_h:.1f} h){clamp_note}",
                         fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
                sp.set_color("#444")
            if ch_i == 0:
                ax.set_ylabel(title, fontsize=12, fontweight="bold", rotation=90,
                              labelpad=10)

        ax_arrow = fig.add_subplot(gs[row_i, 3])
        ax_arrow.text(0.5, 0.5, "→", fontsize=32, ha="center", va="center",
                      transform=ax_arrow.transAxes)
        ax_arrow.axis("off")

        rgb = make_pseudo_rgb(tiff, frame_idx, offsets, fph)
        rgb = center_crop(rgb)
        ax_rgb = fig.add_subplot(gs[row_i, 4])
        ax_rgb.imshow(rgb)
        ax_rgb.set_title(f"Pseudo-RGB\n{note}", fontsize=10)
        ax_rgb.set_xticks([])
        ax_rgb.set_yticks([])
        for sp in ax_rgb.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color("#444")

    fig.suptitle(
        "Temporal pseudo-RGB encoding (zoomed)   —   R = t − 6 h,  G = t − 3 h,  B = t",
        fontsize=13, fontweight="bold", y=1.005,
    )

    for ext in ("png", "pdf"):
        out = OUT / f"fig_temporal_rgb_examples_zoom.{ext}"
        fig.savefig(out)
        print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
