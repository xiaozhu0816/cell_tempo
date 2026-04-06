"""
Generate revised publication-quality Figures 3 and 4 for the Cell Reports paper.

Figure 3: 3-panel layout — binary detection primary, 4-class and MAE as supporting.
Figure 4: 3-panel confusion matrices (overall / early / late) — clean titles.

Outputs saved to: cell_tempo/paper/figures/
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_TMP  = "#1f77b4"   # blue  — temporal
C_SGL  = "#ff7f0e"   # orange — single-frame

BASE    = Path("/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_multiTask")
EP      = "epoch_030"
TEMP_D  = BASE / f"outputs/rowsplit_4cls_temporal_v2/20260327-155528/test_internal/{EP}"
SGLE_D  = BASE / f"outputs/rowsplit_4cls_v2/20260327-203141/test_internal/{EP}"
OUT_DIR = Path("/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo/paper/figures")

CLASS_NAMES = ["MOI 5", "MOI 1", "MOI 0.1", "Mock"]

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
        bk = (h // bw) * bw + bw / 2
        bk = round(bk, 1)
        out.setdefault(bk, []).append(s)
    return dict(sorted(out.items()))

def metrics(samples):
    L  = np.array([s["true_label"] for s in samples])
    P  = np.array([s["pred_label"]  for s in samples])
    bl = (L == 3).astype(int)
    bp = (P == 3).astype(int)
    tt = np.array([s["time_target"] for s in samples])
    tp = np.array([s["time_pred"]   for s in samples])
    return dict(
        acc4 = accuracy_score(L, P),
        bacc = accuracy_score(bl, bp),
        mae  = np.mean(np.abs(tt - tp)),
    )

# ---------------------------------------------------------------------------
# Load and bin
# ---------------------------------------------------------------------------
print("Loading samples...")
t_s = load(TEMP_D)
s_s = load(SGLE_D)

t_bins = bin_hours(t_s)
s_bins = bin_hours(s_s)
common = sorted(set(t_bins) & set(s_bins))

t_met = {h: metrics(t_bins[h]) for h in common}
s_met = {h: metrics(s_bins[h]) for h in common}

# ============================================================================
# FIGURE 3 — 3-panel: (A) Binary  (B) 4-class  (C) MAE
# ============================================================================
print("Generating Figure 3...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                          gridspec_kw={"wspace": 0.32})

PHASE_KW = dict(alpha=0.08)

# ── (A) Binary accuracy — primary, most prominent ──────────────────────────
ax = axes[0]
ax.fill_betweenx([55, 105],  0,  6, color="#e74c3c", **PHASE_KW)
ax.fill_betweenx([55, 105], 24, 48, color="#2980b9", **PHASE_KW)
ax.plot(common, [t_met[h]["bacc"] * 100 for h in common],
        "o-", color=C_TMP, lw=2.5, ms=5.5, zorder=3, label="Temporal")
ax.plot(common, [s_met[h]["bacc"] * 100 for h in common],
        "s--", color=C_SGL, lw=2.5, ms=5.5, zorder=3, label="Single-frame")

ax.set(xlabel="Hours post-infection",
       ylabel="Binary accuracy (%)",
       xlim=(0, 48), ylim=(55, 102))
ax.set_title("(A)  Binary detection accuracy", fontweight="bold", pad=8)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.2, lw=0.5)

phase_handles = [
    Patch(color="#e74c3c", alpha=0.3, label="Early (0–6 h)"),
    Patch(color="#2980b9", alpha=0.3, label="Late (24–48 h)"),
]
ax.legend(handles=[
    Line2D([0],[0], color=C_TMP, lw=2.5, marker="o", ms=5.5, label="Temporal"),
    Line2D([0],[0], color=C_SGL, lw=2.5, marker="s", ms=5.5, ls="--", label="Single-frame"),
] + phase_handles, loc="lower right", fontsize=9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── (B) 4-class accuracy — supporting ──────────────────────────────────────
ax = axes[1]
ax.fill_betweenx([0, 105],  0,  6, color="#e74c3c", **PHASE_KW)
ax.fill_betweenx([0, 105], 24, 48, color="#2980b9", **PHASE_KW)
ax.plot(common, [t_met[h]["acc4"] * 100 for h in common],
        "o-", color=C_TMP, lw=2.5, ms=5.5, zorder=3, label="Temporal")
ax.plot(common, [s_met[h]["acc4"] * 100 for h in common],
        "s--", color=C_SGL, lw=2.5, ms=5.5, zorder=3, label="Single-frame")

ax.set(xlabel="Hours post-infection",
       ylabel="Four-class accuracy (%)",
       xlim=(0, 48), ylim=(0, 100))
ax.set_title("(B)  Severity classification accuracy", fontweight="bold", pad=8)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.2, lw=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── (C) Regression MAE — supporting ────────────────────────────────────────
ax = axes[2]
ax.fill_betweenx([0, 4],  0,  6, color="#e74c3c", **PHASE_KW)
ax.fill_betweenx([0, 4], 24, 48, color="#2980b9", **PHASE_KW)
ax.plot(common, [t_met[h]["mae"] for h in common],
        "o-", color=C_TMP, lw=2.5, ms=5.5, zorder=3, label="Temporal")
ax.plot(common, [s_met[h]["mae"] for h in common],
        "s--", color=C_SGL, lw=2.5, ms=5.5, zorder=3, label="Single-frame")

ax.set(xlabel="Hours post-infection",
       ylabel="MAE (hours)",
       xlim=(0, 48))
ax.set_title("(C)  Time-regression error (MAE)", fontweight="bold", pad=8)
ax.legend(loc="upper left")
ax.grid(True, alpha=0.2, lw=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig1_main_comparison.{ext}", bbox_inches="tight", dpi=200)
plt.close()
print("  Saved Figure 3")

# ============================================================================
# FIGURE 4 — Confusion matrices, clean titles
# ============================================================================
print("Generating Figure 4...")

all_L  = np.array([s["true_label"] for s in t_s])
all_P  = np.array([s["pred_label"]  for s in t_s])
early_s = [s for s in t_s if s["hours"] <  12]
late_s  = [s for s in t_s if s["hours"] >= 24]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.8),
                          gridspec_kw={"wspace": 0.38})

panels = [
    (axes[0], all_L, all_P,        "(A)  Overall"),
    (axes[1],
     np.array([s["true_label"] for s in early_s]),
     np.array([s["pred_label"]  for s in early_s]),
     "(B)  Early phase  (0–12 h)"),
    (axes[2],
     np.array([s["true_label"] for s in late_s]),
     np.array([s["pred_label"]  for s in late_s]),
     "(C)  Late phase  (24–48 h)"),
]

for ax, L, P, title in panels:
    cm = confusion_matrix(L, P)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(4))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(4))
    ax.set_yticklabels(CLASS_NAMES, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontweight="bold", pad=10)

    for i in range(4):
        for j in range(4):
            v = cm_pct[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                    color="white" if v > 55 else "black",
                    fontsize=9.5, fontweight="bold")

    cb = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.03)
    cb.set_label("Row %", fontsize=9)
    cb.ax.tick_params(labelsize=8)

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig2_confusion_matrices.{ext}", bbox_inches="tight", dpi=200)
plt.close()
print("  Saved Figure 4")

print("\nDone. Outputs in:", OUT_DIR)
