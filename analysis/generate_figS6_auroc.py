"""
Figure S6: Per-condition AUROC over time — temporal model only.

Shows AUROC (each infected condition vs Mock) across the 48-hour imaging
timeline for the unified temporal multi-task model, highlighting early
detection onset and sustained reliability.

Three panels: MOI 5 | MOI 1 | MOI 0.1

Output: figS6_auroc_over_time.pdf / .png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PRED_PATH = Path(
    "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_multiTask/outputs"
    "/rowsplit_4cls_temporal_v2/20260327-155528/test_internal/epoch_030/per_sample_results.json"
)
OUT_DIR = Path(
    "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo/paper/supplementary"
)

BIN_WIDTH   = 3     # hours
AUROC_REF   = 0.99  # reference line — matches text claim

MOI_COLORS  = {"moi5": "#D32F2F", "moi1": "#F57C00", "moi01": "#1976D2"}
MOI_LABELS  = {"moi5": "MOI 5", "moi1": "MOI 1", "moi01": "MOI 0.1"}
INFECTED    = ["moi5", "moi1", "moi01"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_samples(path: Path):
    with open(path) as f:
        return json.load(f)


def bin_auroc(samples, moi_condition: str, bin_width: int = 3):
    """Compute AUROC (moi_condition vs mock) per time bin."""
    filtered = [s for s in samples if s["condition"] in (moi_condition, "mock")]
    # Group by bin center
    bins = {}
    for s in filtered:
        bk = round((s["hours"] // bin_width) * bin_width + bin_width / 2, 1)
        bins.setdefault(bk, []).append(s)

    results = {}
    for bk in sorted(bins.keys()):
        bsamps = bins[bk]
        moi_s  = [s for s in bsamps if s["condition"] == moi_condition]
        mock_s = [s for s in bsamps if s["condition"] == "mock"]
        if len(moi_s) < 5 or len(mock_s) < 5:
            continue
        y_true  = [1] * len(moi_s) + [0] * len(mock_s)
        y_score = [1.0 - s["prob_Mock"] for s in moi_s + mock_s]
        try:
            results[bk] = roc_auc_score(y_true, y_score)
        except ValueError:
            pass
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading predictions...")
    samples = load_samples(PRED_PATH)
    print(f"  {len(samples)} samples loaded")

    # ---------------------------------------------------------------------------
    # Plot: 1 row × 3 panels
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)

    for ax, moi in zip(axes, INFECTED):
        per_bin = bin_auroc(samples, moi, BIN_WIDTH)
        times   = sorted(per_bin.keys())
        aurocs  = [per_bin[t] for t in times]

        color = MOI_COLORS[moi]

        # Fill area above reference threshold
        ax.fill_between(times, AUROC_REF, aurocs,
                        where=[a >= AUROC_REF for a in aurocs],
                        color=color, alpha=0.15, interpolate=True)

        # Main AUROC curve
        ax.plot(times, aurocs, "-o", color=color, lw=2.2, ms=3.5,
                markerfacecolor="white", markeredgewidth=1.5)

        # Reference line
        ax.axhline(AUROC_REF, color="gray", ls="--", lw=0.9, alpha=0.7,
                   label=f"AUROC = {AUROC_REF:.2f}")

        ax.set_title(f"{MOI_LABELS[moi]} vs Mock",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Hours post-infection", fontsize=10)
        ax.set_xlim(0, 48)
        ax.set_ylim(0.45, 1.05)
        ax.set_xticks([0, 12, 24, 36, 48])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate earliest time >= AUROC_REF (sustained 2+ consecutive bins)
        sorted_bins = sorted(per_bin.keys())
        earliest = None
        for i, t in enumerate(sorted_bins):
            if per_bin[t] >= AUROC_REF:
                if i + 1 < len(sorted_bins) and per_bin[sorted_bins[i + 1]] >= AUROC_REF:
                    earliest = t
                    break
        if earliest is not None:
            ax.axvline(earliest, color=color, ls=":", lw=1.2, alpha=0.8)
            ax.text(earliest + 0.8, 0.5, f"{earliest:.0f}h",
                    color=color, fontsize=8, va="bottom")

    axes[0].set_ylabel("AUROC (vs Mock)", fontsize=10)
    axes[1].legend(fontsize=8, loc="lower right", framealpha=0.7)

    fig.suptitle(
        "Per-condition detection reliability throughout 48-hour infection timeline\n"
        "Temporal multi-task model",
        fontsize=10, y=1.02
    )
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"figS6_auroc_over_time.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
