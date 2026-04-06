#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "supplementary"


def draw_plate(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    plate = Rectangle((0.6, 0.8), 8.8, 6.4, facecolor="#f6f7f9", edgecolor="#7a8797", linewidth=1.8)
    ax.add_patch(plate)

    col_x = [1.35, 3.35, 5.35, 7.35]
    row_y = [5.4, 3.7, 2.0]
    conds = ["MOI 5", "MOI 1", "MOI 0.1", "Mock"]
    row_labels = ["A/B train", "A/B train", "C test"]
    row_colors = ["#dbeafe", "#dbeafe", "#fee2e2"]

    for x, cond in zip(col_x, conds):
        ax.text(x + 0.75, 7.45, cond, ha="center", va="bottom", fontsize=10, fontweight="bold")

    for y, label, bg in zip(row_y, row_labels, row_colors):
        for x in col_x:
            rect = Rectangle((x, y), 1.5, 1.2, facecolor=bg, edgecolor="#64748b", linewidth=1.0)
            ax.add_patch(rect)
        ax.text(0.2, y + 0.6, label, ha="left", va="center", fontsize=9, color="#334155")

    ax.text(5.0, 0.25, "One biological run = one 12-well plate", ha="center", va="bottom", fontsize=10)


def draw_well_grid(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(5.0, 7.45, "One held-out well", ha="center", va="bottom", fontsize=11, fontweight="bold")

    start_x, start_y = 1.5, 1.1
    cell = 0.85
    for r in range(6):
        for c in range(6):
            rect = Rectangle(
                (start_x + c * cell, start_y + (5 - r) * cell),
                cell * 0.92,
                cell * 0.92,
                facecolor="#e2e8f0" if (r + c) % 2 == 0 else "#cbd5e1",
                edgecolor="white",
                linewidth=0.8,
            )
            ax.add_patch(rect)

    ax.add_patch(Rectangle((start_x - 0.15, start_y - 0.15), 5.25, 5.25, fill=False, edgecolor="#475569", linewidth=1.5))
    ax.text(5.0, 0.45, "6 × 6 acquisition positions = 36 fields of view", ha="center", va="bottom", fontsize=10)


def draw_time_axis(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(5.0, 7.45, "One acquisition position", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(5.0, 6.6, "96 timepoints, every 30 min, over 48 h", ha="center", va="bottom", fontsize=10)

    x0, x1, y = 1.0, 9.0, 3.8
    ax.plot([x0, x1], [y, y], color="#475569", linewidth=2)
    for i in range(0, 97, 12):
        x = x0 + (x1 - x0) * i / 96
        ax.plot([x, x], [y - 0.22, y + 0.22], color="#475569", linewidth=1)

    labels = [("0 h", 0), ("12 h", 24), ("24 h", 48), ("36 h", 72), ("48 h", 96)]
    for text, idx in labels:
        x = x0 + (x1 - x0) * idx / 96
        ax.text(x, y - 0.55, text, ha="center", va="top", fontsize=9)

    ax.text(x0, y + 0.55, "t0", ha="center", va="bottom", fontsize=9)
    ax.text(x1, y + 0.55, "t95", ha="center", va="bottom", fontsize=9)

    current_x = x0 + (x1 - x0) * 74 / 96
    label_specs = [
        (68, "t-6 h", "#ef4444", 1.02),
        (74, "t-3 h", "#22c55e", 0.86),
        (80, "t", "#2563eb", 1.02),
    ]
    for idx, label, color, y_text in label_specs:
        x = x0 + (x1 - x0) * idx / 96
        ax.plot([x, x], [y - 0.4, y + 0.4], color=color, linewidth=2.5)
        ax.text(x, y + y_text, label, ha="center", va="bottom", fontsize=8.8, color=color)
    ax.text(current_x, 1.35, "Three offset frames are sampled\nfrom each position-specific time series", ha="center", va="center", fontsize=9.5)


def draw_rgb_panel(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(5.0, 7.45, "One temporal pseudo-RGB sample", ha="center", va="bottom", fontsize=11, fontweight="bold")

    boxes = [
        (1.0, 3.1, "#fee2e2", "R\n(t-6 h)"),
        (3.2, 3.1, "#dcfce7", "G\n(t-3 h)"),
        (5.4, 3.1, "#dbeafe", "B\n(t)"),
    ]
    for x, y, color, label in boxes:
        rect = Rectangle((x, y), 1.6, 1.8, facecolor=color, edgecolor="#475569", linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + 0.8, y + 0.9, label, ha="center", va="center", fontsize=10, fontweight="bold")

    arrow = FancyArrowPatch((7.15, 4.0), (8.25, 4.0), arrowstyle="simple", mutation_scale=18, color="#64748b")
    ax.add_patch(arrow)
    rgb_rect = Rectangle((8.35, 3.1), 1.0, 1.8, facecolor="#e5e7eb", edgecolor="#475569", linewidth=1.3)
    ax.add_patch(rgb_rect)
    ax.plot([8.35, 9.35], [3.1, 4.9], color="#f97316", linewidth=1.2, alpha=0.6)
    ax.text(8.85, 2.35, "Input to the CNN", ha="center", va="center", fontsize=9.5)
    ax.text(5.0, 1.0, "Hierarchy:\nrun → well → position → timepoint → sample", ha="center", va="center", fontsize=9.2)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    draw_plate(axes[0])
    draw_well_grid(axes[1])
    draw_time_axis(axes[2])
    draw_rgb_panel(axes[3])

    for i in range(3):
        x0 = axes[i].get_position().x1
        x1 = axes[i + 1].get_position().x0
        y = (axes[i].get_position().y0 + axes[i].get_position().y1) / 2
        fig.add_artist(
            FancyArrowPatch(
                (x0 + 0.005, y),
                (x1 - 0.005, y),
                transform=fig.transFigure,
                arrowstyle="simple",
                mutation_scale=14,
                color="#94a3b8",
                alpha=0.9,
            )
        )

    fig.suptitle("Dataset hierarchy and sample construction for temporal brightfield modeling", fontsize=12.5, fontweight="bold", y=0.975)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.84, bottom=0.12, wspace=0.18)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"figS6_dataset_schematic.{ext}", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
