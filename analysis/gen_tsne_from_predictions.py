"""
Fast t-SNE visualization using already-saved prediction logits + per-sample metadata.
No model loading or GPU needed — runs in seconds.

Uses 4-class logits (4-dim) + time prediction (1-dim) = 5D feature space.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "paper" / "figures_new"
EP   = "epoch_030"

TEMPORAL_DIR = ROOT / f"outputs/rowsplit_4cls_temporal_v2/20260327-155528/test_internal/{EP}"
SINGLE_DIR   = ROOT / f"outputs/rowsplit_4cls_v2/20260327-203141/test_internal/{EP}"

CLASS_COLORS = {0: "#e15559", 1: "#ff7f0e", 2: "#2ca02c", 3: "#1f77b4"}
CLASS_NAMES  = {0: "MOI 5", 1: "MOI 1", 2: "MOI 0.1", 3: "Mock"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def load_data(d):
    npz = np.load(d / "predictions.npz")
    meta = json.load(open(d / "per_sample_results.json"))
    return npz, meta


def subsample(npz, meta, n=8000, seed=42):
    rng = np.random.default_rng(seed)
    total = len(npz["labels"])
    idx = rng.choice(total, min(n, total), replace=False)
    idx.sort()
    sub_npz = {k: npz[k][idx] for k in npz.files}
    sub_meta = [meta[i] for i in idx]
    return sub_npz, sub_meta, idx


def run_tsne(features, perplexity=30, seed=42):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                n_iter=1000, init="pca", learning_rate="auto")
    return tsne.fit_transform(features)


def plot_by_class(emb, labels, title, ax):
    for cls_id in sorted(CLASS_COLORS.keys()):
        mask = labels == cls_id
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=CLASS_COLORS[cls_id], label=CLASS_NAMES[cls_id],
                   s=3, alpha=0.4, rasterized=True)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(markerscale=5, fontsize=9, loc="best", framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])


print("Loading prediction data...")
t_npz, t_meta = load_data(TEMPORAL_DIR)
s_npz, s_meta = load_data(SINGLE_DIR)

print("Subsampling...")
t_sub, t_smeta, _ = subsample(t_npz, t_meta)
s_sub, s_smeta, _ = subsample(s_npz, s_meta)

# Build 5D feature vectors: logits(4) + time_pred(1)
t_feats = np.column_stack([t_sub["logits"], t_sub["time_preds"].reshape(-1, 1)])
s_feats = np.column_stack([s_sub["logits"], s_sub["time_preds"].reshape(-1, 1)])

print("Running t-SNE (temporal)...")
emb_t = run_tsne(t_feats)
print("Running t-SNE (single)...")
emb_s = run_tsne(s_feats)

# ── Fig: Side-by-side t-SNE comparison ──────────────────────────
print("Generating figures...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_by_class(emb_t, t_sub["labels"], "Temporal [-6, -3, 0]h", axes[0])
plot_by_class(emb_s, s_sub["labels"], "Single-Frame", axes[1])
fig.suptitle("t-SNE of Prediction Space: Temporal vs Single-Frame (Epoch 30)",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_tsne_temporal_vs_single.png")
fig.savefig(OUT / "fig_tsne_temporal_vs_single.pdf")
plt.close(fig)
print(f"  Saved: fig_tsne_temporal_vs_single")

# ── Fig: Individual t-SNE plots ────────────────────────────────
for emb, sub, smeta, name, tag in [
    (emb_t, t_sub, t_smeta, "Temporal Model", "temporal"),
    (emb_s, s_sub, s_smeta, "Single-Frame Model", "single"),
]:
    fig, ax = plt.subplots(figsize=(7, 6))
    plot_by_class(emb, sub["labels"], f"{name} — Prediction Space (t-SNE)", ax)
    fig.tight_layout()
    fig.savefig(OUT / f"fig_tsne_4cls_{tag}.png")
    fig.savefig(OUT / f"fig_tsne_4cls_{tag}.pdf")
    plt.close(fig)
    print(f"  Saved: fig_tsne_4cls_{tag}")

# ── Fig: t-SNE colored by time ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, emb, smeta, name in [
    (axes[0], emb_t, t_smeta, "Temporal"),
    (axes[1], emb_s, s_smeta, "Single-Frame"),
]:
    hours = np.array([float(m["hours"]) for m in smeta])
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=hours, cmap="viridis",
                    s=3, alpha=0.4, rasterized=True)
    fig.colorbar(sc, ax=ax, label="Hours post-infection", shrink=0.8)
    ax.set_title(f"{name} — Colored by Time", fontsize=12, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("t-SNE Colored by Hours Post-Infection (Epoch 30)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_tsne_by_time.png")
fig.savefig(OUT / "fig_tsne_by_time.pdf")
plt.close(fig)
print(f"  Saved: fig_tsne_by_time")

# ── Fig: t-SNE colored by run ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 6))
run_colors = {"Run2": "#e15559", "Run3": "#1f77b4", "Run4": "#2ca02c"}
runs = []
for m in t_smeta:
    well = m.get("well", "")
    run = well.split("_")[0] if "_" in well else "Unknown"
    runs.append(run)
for run_name, color in run_colors.items():
    mask = np.array([r == run_name for r in runs])
    if mask.any():
        ax.scatter(emb_t[mask, 0], emb_t[mask, 1], c=color, label=run_name,
                   s=3, alpha=0.4, rasterized=True)
ax.set_title("Temporal Model — Colored by Biological Replicate", fontsize=12, fontweight="bold")
ax.legend(markerscale=5, fontsize=10, framealpha=0.9)
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(OUT / "fig_tsne_by_run.png")
fig.savefig(OUT / "fig_tsne_by_run.pdf")
plt.close(fig)
print(f"  Saved: fig_tsne_by_run")

# ── Fig: Correct vs Incorrect ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, emb, sub, name in [
    (axes[0], emb_t, t_sub, "Temporal"),
    (axes[1], emb_s, s_sub, "Single-Frame"),
]:
    correct = sub["preds"] == sub["labels"]
    ax.scatter(emb[correct, 0], emb[correct, 1], c="#2ca02c", label="Correct",
               s=3, alpha=0.3, rasterized=True)
    ax.scatter(emb[~correct, 0], emb[~correct, 1], c="#e15559", label="Incorrect",
               s=5, alpha=0.6, rasterized=True)
    acc = correct.mean() * 100
    ax.set_title(f"{name} — Correct vs Incorrect (Acc={acc:.1f}%)", fontsize=12, fontweight="bold")
    ax.legend(markerscale=5, fontsize=10, framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Prediction Correctness in t-SNE Space (Epoch 30)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_tsne_correctness.png")
fig.savefig(OUT / "fig_tsne_correctness.pdf")
plt.close(fig)
print(f"  Saved: fig_tsne_correctness")

# ── Fig: Temporal phases (early/mid/late) ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
hours = np.array([float(m["hours"]) for m in t_smeta])
phases = [("Early (0-15h)", 0, 15), ("Mid (15-30h)", 15, 30), ("Late (30-48h)", 30, 48)]
for ax, (phase_name, lo, hi) in zip(axes, phases):
    mask_phase = (hours >= lo) & (hours < hi)
    for cls_id in sorted(CLASS_COLORS.keys()):
        mask = mask_phase & (t_sub["labels"] == cls_id)
        ax.scatter(emb_t[mask, 0], emb_t[mask, 1],
                   c=CLASS_COLORS[cls_id], label=CLASS_NAMES[cls_id],
                   s=4, alpha=0.5, rasterized=True)
    n_phase = mask_phase.sum()
    acc_phase = (t_sub["preds"][mask_phase] == t_sub["labels"][mask_phase]).mean() * 100
    ax.set_title(f"{phase_name}\nn={n_phase}, Acc={acc_phase:.1f}%", fontsize=11, fontweight="bold")
    ax.legend(markerscale=4, fontsize=9, framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Temporal Model — Feature Separation by Infection Phase (t-SNE)", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_tsne_temporal_phases.png")
fig.savefig(OUT / "fig_tsne_temporal_phases.pdf")
plt.close(fig)
print(f"  Saved: fig_tsne_temporal_phases")

print("\n=== ALL t-SNE FIGURES DONE ===")
