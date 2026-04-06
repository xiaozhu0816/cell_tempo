"""
Generate per-well accuracy heatmap for the unified temporal 4-class multi-task model.

Wells are organized as a 3×4 grid (biological runs × conditions).
Labels use paper conventions: Run 1, Run 2, Run 3.

Output: figS7_well_heatmap.pdf / .png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METRICS_PATH = Path(
    "/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_multiTask/outputs"
    "/rowsplit_4cls_temporal_v2/20260327-155528/test_internal/epoch_030/per_well_metrics.json"
)
OUT_DIR = Path("/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo/paper/supplementary")

# ---------------------------------------------------------------------------
# Well → (paper_run, condition) mapping
# Internal dataset labels: Run2=paper Run1, Run3=paper Run2, Run4=paper Run3
# ---------------------------------------------------------------------------
WELL_MAP = {
    "Run2_c1": ("Run 1", "MOI 5"),
    "Run2_c2": ("Run 1", "MOI 1"),
    "Run2_c3": ("Run 1", "MOI 0.1"),
    "Run2_c4": ("Run 1", "Mock"),
    "Run3_a2": ("Run 2", "MOI 1"),   # swapped well
    "Run3_c1": ("Run 2", "MOI 5"),
    "Run3_c3": ("Run 2", "MOI 0.1"),
    "Run3_c4": ("Run 2", "Mock"),
    "Run4_a2": ("Run 3", "MOI 5"),   # swapped well
    "Run4_c1": ("Run 3", "MOI 0.1"),
    "Run4_c3": ("Run 3", "MOI 1"),
    "Run4_c4": ("Run 3", "Mock"),
}

RUNS       = ["Run 1", "Run 2", "Run 3"]
CONDITIONS = ["MOI 5", "MOI 1", "MOI 0.1", "Mock"]

# ---------------------------------------------------------------------------
# Load and organise
# ---------------------------------------------------------------------------
with open(METRICS_PATH) as f:
    data = json.load(f)["per_well"]

# Build 3×4 accuracy grid
acc_grid = np.full((len(RUNS), len(CONDITIONS)), np.nan)
for well_id, (run, cond) in WELL_MAP.items():
    if well_id in data:
        r = RUNS.index(run)
        c = CONDITIONS.index(cond)
        acc_grid[r, c] = data[well_id]["accuracy"]

print("Accuracy grid (Run × Condition):")
for i, run in enumerate(RUNS):
    for j, cond in enumerate(CONDITIONS):
        print(f"  {run} / {cond}: {acc_grid[i,j]:.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.2))

# Color map: red (low) → white (mid) → green (high)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "rg", ["#d62728", "#f5f5f5", "#2ca02c"], N=256
)
im = ax.imshow(acc_grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

# Annotate each cell with the accuracy value
for r in range(len(RUNS)):
    for c in range(len(CONDITIONS)):
        val = acc_grid[r, c]
        if not np.isnan(val):
            # Use dark text on light cells, light text on dark cells
            text_color = "black" if 0.35 < val < 0.85 else "white"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

# Axes labels
ax.set_xticks(range(len(CONDITIONS)))
ax.set_xticklabels(CONDITIONS, fontsize=11)
ax.set_yticks(range(len(RUNS)))
ax.set_yticklabels(RUNS, fontsize=11)
ax.set_xlabel("Infection condition", fontsize=12, labelpad=8)
ax.set_ylabel("Biological replicate", fontsize=12, labelpad=8)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
cbar.set_label("Per-well 4-class accuracy", fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Grid lines between cells
ax.set_xticks(np.arange(-0.5, len(CONDITIONS)), minor=True)
ax.set_yticks(np.arange(-0.5, len(RUNS)), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", bottom=False, left=False)

fig.suptitle(
    "Per-well classification accuracy — temporal multi-task model",
    fontsize=11, y=1.01
)
fig.tight_layout()

# Save
for ext in ("pdf", "png"):
    out = OUT_DIR / f"figS7_well_heatmap.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")

plt.close()
