"""
Extract ResNet50 backbone features from trained models and generate t-SNE visualizations.

Generates:
  - fig_tsne_4cls_temporal.png/pdf     : t-SNE colored by 4-class label (temporal model)
  - fig_tsne_4cls_single.png/pdf       : t-SNE colored by 4-class label (single model)
  - fig_tsne_temporal_vs_single.png/pdf : side-by-side comparison
  - fig_tsne_by_time.png/pdf           : t-SNE colored by time-since-infection
  - fig_tsne_by_run.png/pdf            : t-SNE colored by biological replicate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.run2_dataset import build_row_split_dataset
from models import build_multitask_model
from utils import load_config, set_seed, build_transforms

OUT = ROOT / "paper" / "figures_new"
OUT.mkdir(parents=True, exist_ok=True)

set_seed(42)

# ── Style ────────────────────────────────────────────────────────────
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

CLASS_COLORS = {0: "#e15559", 1: "#ff7f0e", 2: "#2ca02c", 3: "#1f77b4"}
CLASS_NAMES  = {0: "MOI 5", 1: "MOI 1", 2: "MOI 0.1", 3: "Mock"}


def extract_features(model, loader, device, max_samples=10000):
    """Extract backbone features, labels, metadata from a model.
    Extracts ALL samples first, then stratified-subsamples to max_samples
    so every class is represented proportionally.
    """
    model.eval()
    all_feats, all_labels, all_meta = [], [], []
    with torch.no_grad():
        for imgs, labs, meta in tqdm(loader, desc="Extracting features"):
            imgs = imgs.to(device, non_blocking=True)
            feats = model.get_features(imgs)  # [B, 2048]
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labs.numpy())
            # Convert batched meta to list of dicts
            if isinstance(meta, dict):
                bs = imgs.size(0)
                for i in range(bs):
                    m = {k: (v[i].item() if torch.is_tensor(v[i]) else v[i]) for k, v in meta.items()}
                    all_meta.append(m)
            elif isinstance(meta, list):
                all_meta.extend(meta)

    feats = np.concatenate(all_feats)
    labels = np.concatenate(all_labels)
    meta_list = all_meta

    # Stratified subsample to max_samples (keep class proportions)
    if len(labels) > max_samples:
        rng = np.random.default_rng(42)
        unique_cls = np.unique(labels)
        selected = []
        for cls_id in unique_cls:
            cls_idx = np.where(labels == cls_id)[0]
            n_take = max(1, int(max_samples * len(cls_idx) / len(labels)))
            chosen = rng.choice(cls_idx, min(n_take, len(cls_idx)), replace=False)
            selected.append(chosen)
        selected = np.concatenate(selected)
        rng.shuffle(selected)
        selected = selected[:max_samples]
        selected.sort()
        feats = feats[selected]
        labels = labels[selected]
        meta_list = [meta_list[i] for i in selected]
        print(f"  Subsampled: {dict(zip(*np.unique(labels, return_counts=True)))}")

    return feats, labels, meta_list


def run_tsne(feats, perplexity=30, seed=42):
    """Run t-SNE on features."""
    print(f"Running t-SNE on {feats.shape[0]} samples, dim={feats.shape[1]}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                n_iter=1000, init="pca", learning_rate="auto")
    emb = tsne.fit_transform(feats)
    print(f"  KL divergence: {tsne.kl_divergence_:.4f}")
    return emb


def plot_tsne_by_class(emb, labels, title, save_path, ax=None):
    """Plot t-SNE colored by class label."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 6))

    for cls_id in sorted(CLASS_COLORS.keys()):
        mask = labels == cls_id
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=CLASS_COLORS[cls_id], label=CLASS_NAMES[cls_id],
                   s=3, alpha=0.4, rasterized=True)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(markerscale=5, fontsize=10, loc="best", framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])

    if standalone:
        fig.tight_layout()
        fig.savefig(save_path.with_suffix(".png"))
        fig.savefig(save_path.with_suffix(".pdf"))
        plt.close(fig)
        print(f"  Saved: {save_path}")


def plot_tsne_by_time(emb, meta_list, title, save_path):
    """Plot t-SNE colored by hours since infection."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    hours = np.array([float(m.get("hours", m.get("frame_index", 0)) if isinstance(m, dict) else 0) for m in meta_list])
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=hours, cmap="viridis",
                    s=3, alpha=0.4, rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, label="Hours post-infection")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path.with_suffix(".png"))
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_tsne_by_run(emb, meta_list, title, save_path):
    """Plot t-SNE colored by biological replicate (Run)."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    run_colors = {"Run2": "#e15559", "Run3": "#1f77b4", "Run4": "#2ca02c"}
    for m in meta_list:
        well = m.get("well", "")
        if isinstance(well, str):
            run = well.split("_")[0] if "_" in well else "Unknown"
        else:
            run = "Unknown"
        m["_run"] = run

    runs = [m["_run"] for m in meta_list]
    for run_name, color in run_colors.items():
        mask = np.array([r == run_name for r in runs])
        if mask.any():
            ax.scatter(emb[mask, 0], emb[mask, 1], c=color, label=run_name,
                       s=3, alpha=0.4, rasterized=True)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(markerscale=5, fontsize=10, loc="best", framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path.with_suffix(".png"))
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_tsne_by_correctness(emb, labels, preds, title, save_path):
    """Plot t-SNE colored by correct/incorrect prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    correct = (labels == preds)

    # Left: colored by class
    ax = axes[0]
    for cls_id in sorted(CLASS_COLORS.keys()):
        mask = labels == cls_id
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=CLASS_COLORS[cls_id], label=CLASS_NAMES[cls_id],
                   s=3, alpha=0.4, rasterized=True)
    ax.set_title(f"{title} — by Class")
    ax.legend(markerscale=5, fontsize=10, framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])

    # Right: correct vs incorrect
    ax = axes[1]
    ax.scatter(emb[correct, 0], emb[correct, 1], c="#2ca02c", label="Correct",
               s=3, alpha=0.3, rasterized=True)
    ax.scatter(emb[~correct, 0], emb[~correct, 1], c="#e15559", label="Incorrect",
               s=5, alpha=0.6, rasterized=True)
    ax.set_title(f"{title} — Correct vs Incorrect")
    ax.legend(markerscale=5, fontsize=10, framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(save_path.with_suffix(".png"))
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_tsne_temporal_phases(emb, meta_list, labels, title, save_path):
    """Plot t-SNE as 3 panels: early (0-15h), mid (15-30h), late (30-48h)."""
    hours = np.array([float(m.get("hours", 0)) for m in meta_list])
    phases = [("Early (0–15h)", (0, 15)), ("Mid (15–30h)", (15, 30)), ("Late (30–48h)", (30, 48))]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (phase_name, (lo, hi)) in zip(axes, phases):
        mask_phase = (hours >= lo) & (hours < hi)
        for cls_id in sorted(CLASS_COLORS.keys()):
            mask = mask_phase & (labels == cls_id)
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=CLASS_COLORS[cls_id], label=CLASS_NAMES[cls_id],
                       s=4, alpha=0.5, rasterized=True)
        ax.set_title(phase_name)
        ax.legend(markerscale=4, fontsize=9, framealpha=0.9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path.with_suffix(".png"))
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load configs ─────────────────────────────────────────────────
    cfg_temporal = load_config(ROOT / "configs" / "rowsplit_4cls_temporal_v2.yaml")
    cfg_single   = load_config(ROOT / "configs" / "rowsplit_4cls_v2.yaml")

    # ── Checkpoints ──────────────────────────────────────────────────
    ckpt_temporal = ROOT / "outputs/rowsplit_4cls_temporal_v2/20260327-155528/checkpoints/epoch_030.pt"
    ckpt_single   = ROOT / "outputs/rowsplit_4cls_v2/20260327-203141/checkpoints/epoch_030.pt"

    # ── Build datasets (test split only) ─────────────────────────────
    print("\n=== Building temporal dataset ===")
    tx_temporal = build_transforms(cfg_temporal["data"]["transforms"])
    ds_temporal = build_row_split_dataset(cfg_temporal["data"], tx_temporal)
    test_loader_temporal = DataLoader(
        ds_temporal["test_internal"],
        batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )

    print("\n=== Building single-frame dataset ===")
    tx_single = build_transforms(cfg_single["data"]["transforms"])
    ds_single = build_row_split_dataset(cfg_single["data"], tx_single)
    test_loader_single = DataLoader(
        ds_single["test_internal"],
        batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )

    # ── Load models ──────────────────────────────────────────────────
    print("\n=== Loading temporal model ===")
    model_temporal = build_multitask_model(cfg_temporal["model"]).to(device)
    state = torch.load(ckpt_temporal, map_location=device)
    model_temporal.load_state_dict(state.get("model_state", state.get("model_state_dict", state)))
    model_temporal.eval()

    print("=== Loading single-frame model ===")
    model_single = build_multitask_model(cfg_single["model"]).to(device)
    state = torch.load(ckpt_single, map_location=device)
    model_single.load_state_dict(state.get("model_state", state.get("model_state_dict", state)))
    model_single.eval()

    MAX_SAMPLES = 8000  # subsample for t-SNE speed + memory

    # ── Extract features ─────────────────────────────────────────────
    print("\n=== Extracting TEMPORAL features ===")
    feats_t, labels_t, meta_t = extract_features(model_temporal, test_loader_temporal, device, MAX_SAMPLES)

    print("\n=== Extracting SINGLE features ===")
    feats_s, labels_s, meta_s = extract_features(model_single, test_loader_single, device, MAX_SAMPLES)

    # Save features for reuse
    np.savez(OUT / "tsne_features_temporal.npz", feats=feats_t, labels=labels_t)
    np.savez(OUT / "tsne_features_single.npz", feats=feats_s, labels=labels_s)

    # ── t-SNE ────────────────────────────────────────────────────────
    print("\n=== Running t-SNE (temporal) ===")
    emb_t = run_tsne(feats_t)

    print("\n=== Running t-SNE (single) ===")
    emb_s = run_tsne(feats_s)

    # Also get predictions for correctness plot
    npz_t = np.load(ROOT / "outputs/rowsplit_4cls_temporal_v2/20260327-155528/test_internal/epoch_030/predictions.npz")
    npz_s = np.load(ROOT / "outputs/rowsplit_4cls_v2/20260327-203141/test_internal/epoch_030/predictions.npz")
    preds_t = npz_t["preds"][:MAX_SAMPLES]
    preds_s = npz_s["preds"][:MAX_SAMPLES]

    # ── Generate figures ─────────────────────────────────────────────
    print("\n=== Generating figures ===")

    # Fig: t-SNE by class (temporal)
    plot_tsne_by_class(emb_t, labels_t,
                       "Temporal Model — Feature Space (t-SNE)",
                       OUT / "fig_tsne_4cls_temporal")

    # Fig: t-SNE by class (single)
    plot_tsne_by_class(emb_s, labels_s,
                       "Single-Frame Model — Feature Space (t-SNE)",
                       OUT / "fig_tsne_4cls_single")

    # Fig: Side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_tsne_by_class(emb_t, labels_t, "Temporal [-6, -3, 0]h", None, ax=axes[0])
    plot_tsne_by_class(emb_s, labels_s, "Single-Frame", None, ax=axes[1])
    fig.suptitle("t-SNE Feature Visualization: Temporal vs Single-Frame (Epoch 30)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tsne_temporal_vs_single.png")
    fig.savefig(OUT / "fig_tsne_temporal_vs_single.pdf")
    plt.close(fig)
    print(f"  Saved: fig_tsne_temporal_vs_single")

    # Fig: t-SNE by time
    plot_tsne_by_time(emb_t, meta_t,
                      "Temporal Model — Colored by Hours Post-Infection",
                      OUT / "fig_tsne_by_time")

    # Fig: t-SNE by run
    plot_tsne_by_run(emb_t, meta_t,
                     "Temporal Model — Colored by Biological Replicate",
                     OUT / "fig_tsne_by_run")

    # Fig: Correct vs Incorrect
    plot_tsne_by_correctness(emb_t, labels_t, preds_t,
                             "Temporal Model",
                             OUT / "fig_tsne_correctness_temporal")

    plot_tsne_by_correctness(emb_s, labels_s, preds_s,
                             "Single-Frame Model",
                             OUT / "fig_tsne_correctness_single")

    # Fig: Temporal phases
    plot_tsne_temporal_phases(emb_t, meta_t, labels_t,
                             "Temporal Model — Feature Separation by Infection Phase",
                             OUT / "fig_tsne_temporal_phases")

    print("\n=== ALL t-SNE FIGURES DONE ===")


if __name__ == "__main__":
    main()
