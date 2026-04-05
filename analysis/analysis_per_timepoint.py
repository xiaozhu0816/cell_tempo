"""Per-timepoint analysis: compare temporal vs single model accuracy over infection hours."""
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# ── Paths ──
TEMPORAL_DIR = pathlib.Path(
    "outputs/rowsplit_4cls_temporal_v2/20260327-155528/test_internal/epoch_030"
)
SINGLE_DIR = pathlib.Path(
    "outputs/rowsplit_4cls_v2/20260327-203141/test_internal/epoch_025"
)
OUT_DIR = pathlib.Path("analysis_per_timepoint")
OUT_DIR.mkdir(exist_ok=True)

CLASS_NAMES_4 = ["MOI5", "MOI1", "MOI0.1", "Mock"]
BIN_NAMES = ["Infected", "Mock"]

def load_samples(d):
    with open(d / "per_sample_results.json") as f:
        return json.load(f)

def bin_by_hours(samples, bin_width=3):
    """Bin samples by hours, return dict of bin_center -> list of samples."""
    hours = np.array([s["hours"] for s in samples])
    hmin, hmax = hours.min(), hours.max()
    edges = np.arange(hmin, hmax + bin_width, bin_width)
    bins = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        center = (lo + hi) / 2
        subset = [s for s in samples if lo <= s["hours"] < hi]
        if subset:
            bins[center] = subset
    return bins

def compute_metrics(samples):
    labels = np.array([s["true_label"] for s in samples])
    preds = np.array([s["pred_label"] for s in samples])
    acc4 = accuracy_score(labels, preds)
    f1_4 = f1_score(labels, preds, average="macro", zero_division=0)
    # binary
    mock_idx = 3
    bl = (labels == mock_idx).astype(int)
    bp = (preds == mock_idx).astype(int)
    acc_bin = accuracy_score(bl, bp)
    f1_bin = f1_score(bl, bp, average="macro", zero_division=0)
    # regression
    tp = np.array([s["time_pred"] for s in samples])
    tt = np.array([s["time_target"] for s in samples])
    mae = np.mean(np.abs(tp - tt))
    return dict(acc4=acc4, f1_4=f1_4, acc_bin=acc_bin, f1_bin=f1_bin, mae=mae, n=len(samples))

def per_condition_timepoint(samples, bin_width=3):
    """Per condition + timepoint breakdown."""
    conditions = sorted(set(s["condition"] for s in samples))
    result = {}
    for cond in conditions:
        csamps = [s for s in samples if s["condition"] == cond]
        bins = bin_by_hours(csamps, bin_width)
        result[cond] = {h: compute_metrics(subs) for h, subs in sorted(bins.items())}
    return result

print("Loading temporal model (E30) ...")
t_samples = load_samples(TEMPORAL_DIR)
print(f"  {len(t_samples)} samples")

print("Loading single model (E25) ...")
s_samples = load_samples(SINGLE_DIR)
print(f"  {len(s_samples)} samples")

BIN_W = 3  # 3-hour bins

# ── Overall per-timepoint ──
t_bins = bin_by_hours(t_samples, BIN_W)
s_bins = bin_by_hours(s_samples, BIN_W)

hours_common = sorted(set(t_bins.keys()) & set(s_bins.keys()))

t_met = {h: compute_metrics(t_bins[h]) for h in hours_common}
s_met = {h: compute_metrics(s_bins[h]) for h in hours_common}

# ── Print table ──
print(f"\n{'Hour':>6} | {'Temporal Acc4':>13} {'Single Acc4':>12} {'Δ':>6} | "
      f"{'Temporal BinAcc':>15} {'Single BinAcc':>14} {'Δ':>6} | "
      f"{'Temporal MAE':>12} {'Single MAE':>11} | N")
print("-" * 120)
for h in hours_common:
    tm, sm = t_met[h], s_met[h]
    d4 = (tm["acc4"] - sm["acc4"]) * 100
    db = (tm["acc_bin"] - sm["acc_bin"]) * 100
    print(f"{h:6.1f} | {tm['acc4']*100:12.1f}% {sm['acc4']*100:11.1f}% {d4:+5.1f} | "
          f"{tm['acc_bin']*100:14.1f}% {sm['acc_bin']*100:13.1f}% {db:+5.1f} | "
          f"{tm['mae']:11.2f}h {sm['mae']:10.2f}h | {tm['n']}")

# ── Per-condition per-timepoint ──
print("\n\n=== Per-condition per-timepoint (Temporal model E30) ===")
t_cond = per_condition_timepoint(t_samples, BIN_W)
for cond in sorted(t_cond.keys()):
    print(f"\n  {cond}:")
    for h, m in sorted(t_cond[cond].items()):
        print(f"    {h:5.1f}h: acc4={m['acc4']*100:5.1f}%  f1={m['f1_4']:.3f}  MAE={m['mae']:.2f}h  (n={m['n']})")

# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ── 1. 4-class accuracy over time ──
ax = axes[0, 0]
ax.plot(hours_common, [t_met[h]["acc4"] * 100 for h in hours_common],
        "o-", label="Temporal", color="tab:blue", linewidth=2)
ax.plot(hours_common, [s_met[h]["acc4"] * 100 for h in hours_common],
        "s--", label="Single", color="tab:orange", linewidth=2)
ax.set_xlabel("Hours post-infection")
ax.set_ylabel("4-class Accuracy (%)")
ax.set_title("4-class Accuracy over Time")
ax.legend()
ax.grid(True, alpha=0.3)

# ── 2. Binary accuracy over time ──
ax = axes[0, 1]
ax.plot(hours_common, [t_met[h]["acc_bin"] * 100 for h in hours_common],
        "o-", label="Temporal", color="tab:blue", linewidth=2)
ax.plot(hours_common, [s_met[h]["acc_bin"] * 100 for h in hours_common],
        "s--", label="Single", color="tab:orange", linewidth=2)
ax.set_xlabel("Hours post-infection")
ax.set_ylabel("Binary Accuracy (%)")
ax.set_title("Binary (Infected vs Mock) Accuracy over Time")
ax.legend()
ax.grid(True, alpha=0.3)

# ── 3. Regression MAE over time ──
ax = axes[1, 0]
ax.plot(hours_common, [t_met[h]["mae"] for h in hours_common],
        "o-", label="Temporal", color="tab:blue", linewidth=2)
ax.plot(hours_common, [s_met[h]["mae"] for h in hours_common],
        "s--", label="Single", color="tab:orange", linewidth=2)
ax.set_xlabel("Hours post-infection")
ax.set_ylabel("MAE (hours)")
ax.set_title("Regression MAE over Time")
ax.legend()
ax.grid(True, alpha=0.3)

# ── 4. Temporal advantage (Δ accuracy) ──
ax = axes[1, 1]
delta4 = [(t_met[h]["acc4"] - s_met[h]["acc4"]) * 100 for h in hours_common]
delta_bin = [(t_met[h]["acc_bin"] - s_met[h]["acc_bin"]) * 100 for h in hours_common]
ax.bar([h - 0.5 for h in hours_common], delta4, width=1.0, alpha=0.7,
       label="Δ 4cls Acc", color="tab:blue")
ax.bar([h + 0.5 for h in hours_common], delta_bin, width=1.0, alpha=0.7,
       label="Δ Binary Acc", color="tab:green")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Hours post-infection")
ax.set_ylabel("Temporal − Single (pp)")
ax.set_title("Temporal Advantage over Time")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "temporal_vs_single_over_time.png", dpi=200)
fig.savefig(OUT_DIR / "temporal_vs_single_over_time.pdf")
print(f"\nSaved: {OUT_DIR / 'temporal_vs_single_over_time.png'}")

# ── Per-condition accuracy curves (temporal only) ──
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
conditions = sorted(t_cond.keys())
colors = {"moi5": "red", "moi1": "orange", "moi01": "green", "mock": "blue"}

ax = axes2[0]
for cond in conditions:
    hrs = sorted(t_cond[cond].keys())
    accs = [t_cond[cond][h]["acc4"] * 100 for h in hrs]
    ax.plot(hrs, accs, "o-", label=cond, color=colors.get(cond, "gray"), linewidth=2)
ax.set_xlabel("Hours post-infection")
ax.set_ylabel("4-class Accuracy (%)")
ax.set_title("Per-condition Accuracy (Temporal model)")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes2[1]
for cond in conditions:
    hrs = sorted(t_cond[cond].keys())
    maes = [t_cond[cond][h]["mae"] for h in hrs]
    ax.plot(hrs, maes, "o-", label=cond, color=colors.get(cond, "gray"), linewidth=2)
ax.set_xlabel("Hours post-infection")
ax.set_ylabel("MAE (hours)")
ax.set_title("Per-condition Regression MAE (Temporal model)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig(OUT_DIR / "per_condition_over_time.png", dpi=200)
fig2.savefig(OUT_DIR / "per_condition_over_time.pdf")
print(f"Saved: {OUT_DIR / 'per_condition_over_time.png'}")

# ── Early vs Late analysis ──
print("\n\n=== Early (0-12h) vs Mid (12-24h) vs Late (24-48h) ===")
for name, model_samples in [("Temporal", t_samples), ("Single", s_samples)]:
    print(f"\n  {name}:")
    for label, lo, hi in [("Early 0-12h", 0, 12), ("Mid 12-24h", 12, 24), ("Late 24-48h", 24, 48)]:
        subset = [s for s in model_samples if lo <= s["hours"] < hi]
        if subset:
            m = compute_metrics(subset)
            print(f"    {label}: acc4={m['acc4']*100:.1f}%  f1={m['f1_4']:.3f}  "
                  f"binAcc={m['acc_bin']*100:.1f}%  MAE={m['mae']:.2f}h  (n={m['n']})")

# ── Infected-only early detection analysis ──
print("\n\n=== Early detection (Infected only, 0-12h): Temporal vs Single ===")
for name, model_samples in [("Temporal", t_samples), ("Single", s_samples)]:
    inf_early = [s for s in model_samples if s["condition"] != "mock" and s["hours"] < 12]
    if inf_early:
        labels = np.array([s["true_label"] for s in inf_early])
        preds = np.array([s["pred_label"] for s in inf_early])
        # binary: did the model detect it as infected (not mock)?
        bl = np.zeros(len(labels))  # all should be infected = 0
        bp = (preds == 3).astype(int)  # 1 if predicted mock (false negative)
        fn_rate = bp.mean() * 100
        # 4cls accuracy
        acc4 = accuracy_score(labels, preds)
        print(f"  {name}: 4cls_acc={acc4*100:.1f}%  "
              f"False_negative_rate={fn_rate:.1f}%  (n={len(inf_early)})")

print("\nDone.")
