"""
Generate cleaned-up supplementary Figures S4 and S5.

Figure S4: Predicted infection probability (P(Infected)) over time for
           temporal and single-frame models by condition.

Figure S5: Per-time-window confusion matrices for temporal and single-frame
           models (2 rows × 3 columns: Temporal / Single-frame × 0-12h / 12-24h / 24-48h).

Outputs saved to: cell_tempo/paper/supplementary/
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BASE   = Path("/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_multiTask")
EP     = "epoch_030"
TEMP_D = BASE / f"outputs/rowsplit_4cls_temporal_v2/20260327-155528/test_internal/{EP}"
SGLE_D = BASE / f"outputs/rowsplit_4cls_v2/20260327-203141/test_internal/{EP}"
OUT    = Path("/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo/paper/supplementary")

CLASS_NAMES = ["MOI 5", "MOI 1", "MOI 0.1", "Mock"]

COND_COLORS = {
    "moi5":  "#e15759",
    "moi1":  "#f28e2b",
    "moi01": "#59a14f",
    "mock":  "#4e79a7",
}
COND_LABELS = {
    "moi5":  "MOI 5",
    "moi1":  "MOI 1",
    "moi01": "MOI 0.1",
    "mock":  "Mock",
}
COND_ORDER = ["moi5", "moi1", "moi01", "mock"]

C_TMP = "#1f77b4"
C_SGL = "#ff7f0e"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path):
    with open(path / "per_sample_results.json") as f:
        return json.load(f)

def bin_hours(samples, bw=3):
    out = {}
    for s in samples:
        h = s["hours"]
        bk = round((h // bw) * bw + bw / 2, 1)
        out.setdefault(bk, []).append(s)
    return dict(sorted(out.items()))

def p_infected(samples):
    """Mean P(Infected) = 1 - P(Mock) per sample list."""
    return np.mean([1.0 - s["prob_Mock"] for s in samples])

def p_infected_sem(samples):
    vals = np.array([1.0 - s["prob_Mock"] for s in samples])
    return vals.std() / np.sqrt(len(vals))

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading samples...")
t_s = load(TEMP_D)
s_s = load(SGLE_D)

# ============================================================================
# FIGURE S4 — Confidence trajectories (P(Infected) over time)
# ============================================================================
print("Generating Figure S4...")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                          gridspec_kw={"wspace": 0.28})

for ax, samples, model_label in [
    (axes[0], t_s, "Temporal model"),
    (axes[1], s_s, "Single-frame model"),
]:
    by_cond = {}
    for s in samples:
        by_cond.setdefault(s["condition"], []).append(s)

    for cond in COND_ORDER:
        cs = by_cond.get(cond, [])
        if not cs:
            continue
        binned = bin_hours(cs)
        hrs  = sorted(binned.keys())
        vals = [p_infected(binned[h]) for h in hrs]
        sems = [p_infected_sem(binned[h]) for h in hrs]
        col  = COND_COLORS[cond]
        ax.plot(hrs, vals, "-", color=col, lw=2.2,
                label=COND_LABELS[cond])
        ax.fill_between(hrs,
                        [v - s for v, s in zip(vals, sems)],
                        [v + s for v, s in zip(vals, sems)],
                        color=col, alpha=0.12)

    ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_title(model_label, fontweight="bold", pad=8)
    ax.set_xlabel("Hours post-infection")
    ax.set_ylabel("P(Infected)")
    ax.set_xlim(0, 48)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="center right", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()
for ext in ("pdf", "png"):
    out = OUT / f"figS4_confidence_trajectories.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"  Saved: {out}")
plt.close()

# ============================================================================
# FIGURE S5 — Per-window confusion matrices (Temporal vs Single-frame)
# ============================================================================
print("Generating Figure S5...")

windows = [
    ("0–12 h",  lambda s: s["hours"] <  12),
    ("12–24 h", lambda s: 12 <= s["hours"] < 24),
    ("24–48 h", lambda s: s["hours"] >= 24),
]

# 2 rows (models) × 3 cols (windows) + 1 narrow colorbar col
fig = plt.figure(figsize=(13, 8.0))
gs = mgridspec.GridSpec(2, 4,
                        width_ratios=[1, 1, 1, 0.055],
                        wspace=0.28, hspace=0.40,
                        figure=fig)

row_labels = ["Temporal model", "Single-frame model"]
last_im = None

for row_i, (samples, row_label) in enumerate([
    (t_s, "Temporal model"),
    (s_s, "Single-frame model"),
]):
    for col_i, (win_label, win_fn) in enumerate(windows):
        ax = fig.add_subplot(gs[row_i, col_i])
        sub = [s for s in samples if win_fn(s)]

        L = np.array([s["true_label"] for s in sub])
        P = np.array([s["pred_label"]  for s in sub])
        cm = confusion_matrix(L, P)
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        last_im = im

        ax.set_xticks(range(4))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        if col_i == 0:
            ax.set_ylabel("True", fontsize=9)

        # Column header on top row only
        if row_i == 0:
            ax.set_title(win_label, fontweight="bold", pad=8, fontsize=11)

        # Row label on leftmost column only
        if col_i == 0:
            ax.text(-0.28, 0.5, row_label, transform=ax.transAxes,
                    fontsize=10, fontweight="bold", rotation=90,
                    va="center", ha="center")

        for i in range(4):
            for j in range(4):
                v = cm_pct[i, j]
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        color="white" if v > 55 else "black",
                        fontsize=8, fontweight="bold")

cax = fig.add_subplot(gs[:, 3])
cb  = fig.colorbar(last_im, cax=cax)
cb.set_label("Row %", fontsize=9)
cb.ax.tick_params(labelsize=8)

for ext in ("pdf", "png"):
    out = OUT / f"figS5_confusion_per_phase.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"  Saved: {out}")
plt.close()

print("\nDone.")
