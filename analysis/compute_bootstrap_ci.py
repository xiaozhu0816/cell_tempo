"""
Bootstrap 95% confidence intervals for Table 1 and Table 2 metrics.
Resamples at the well level (12 test wells) for biologically meaningful CIs.
Also computes McNemar's test for the key temporal vs single-frame binary comparison.

Usage:
    python compute_bootstrap_ci.py

Outputs:
    bootstrap_ci_table1.csv
    bootstrap_ci_table2.csv
    mcnemar_result.txt
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import csv

BASE = Path('/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_multiTask/outputs')
OUT_DIR = Path('/isilon/datalake/gurcan_rsch/scratch/WSI/zhengjie/CODE/cell_tempo/analysis')

N_BOOTSTRAP = 1000
SEED = 42
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_samples(path):
    """Load per_sample_results.json → list of dicts."""
    with open(path) as f:
        return json.load(f)


def derive_binary(pred_label, n_classes=4):
    """Collapse 4-class prediction to binary: classes 0-2 → Infected (0), class 3 → Mock (1)."""
    return 0 if pred_label < (n_classes - 1) else 1


def group_by_well(samples):
    """Return dict: well_id → list of sample dicts."""
    wells = defaultdict(list)
    for s in samples:
        wells[s['well']].append(s)
    return wells


def compute_metrics(samples, task='binary', has_reg=True):
    """
    Compute accuracy, macro-F1, MAE, R^2 from a flat list of samples.

    task:
      'binary'  - samples have pred_label/true_label as 0/1 directly
      '4cls'    - samples have pred_label/true_label as 0-3; binary derived by collapsing
      '4cls_binary' - like '4cls' but returns BINARY accuracy (derived)
    """
    if not samples:
        return {}

    # Classification metrics
    if task == 'binary':
        true = np.array([s['true_label'] for s in samples])
        pred = np.array([s['pred_label'] for s in samples])
    elif task in ('4cls', '4cls_binary'):
        true_4 = np.array([s['true_label'] for s in samples])
        pred_4 = np.array([s['pred_label'] for s in samples])
        if task == '4cls':
            true = true_4
            pred = pred_4
        else:  # 4cls_binary: derive binary
            true = np.array([0 if t < 3 else 1 for t in true_4])
            pred = np.array([0 if p < 3 else 1 for p in pred_4])

    accuracy = np.mean(true == pred)

    # Macro F1
    classes = np.unique(true)
    f1s = []
    for c in classes:
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = np.mean(f1s)

    metrics = {'accuracy': accuracy, 'f1_macro': macro_f1}

    # Regression metrics
    if has_reg:
        time_true = np.array([s['time_target'] for s in samples])
        time_pred = np.array([s['time_pred'] for s in samples])
        mae = np.mean(np.abs(time_true - time_pred))
        ss_res = np.sum((time_true - time_pred) ** 2)
        ss_tot = np.sum((time_true - np.mean(time_true)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        metrics['mae'] = mae
        metrics['r2'] = r2

    return metrics


def bootstrap_ci(samples, task='binary', has_reg=True, n=N_BOOTSTRAP):
    """
    Bootstrap 95% CI by resampling wells with replacement.
    Returns dict: metric → (lower_95, upper_95)
    """
    wells = group_by_well(samples)
    well_ids = list(wells.keys())
    k = len(well_ids)

    boot_results = defaultdict(list)
    for _ in range(n):
        sampled_wells = rng.choice(well_ids, size=k, replace=True)
        boot_samples = []
        for w in sampled_wells:
            boot_samples.extend(wells[w])
        m = compute_metrics(boot_samples, task=task, has_reg=has_reg)
        for key, val in m.items():
            boot_results[key].append(val)

    ci = {}
    for key, vals in boot_results.items():
        arr = np.array(vals)
        ci[key] = (np.percentile(arr, 2.5), np.percentile(arr, 97.5))
    return ci


# ---------------------------------------------------------------------------
# Experiment registry — maps paper row → (path, task, has_reg)
# ---------------------------------------------------------------------------

def exp_path(name, ts):
    return BASE / name / ts / 'test_internal' / 'epoch_030' / 'per_sample_results.json'


TABLE1_EXPERIMENTS = [
    # (row_label, path, task_for_reported_metric, has_reg)
    # SF models
    ('SF binary cls-only',
     exp_path('rowsplit_binary_v2_cls_only', '20260401-151750'), 'binary', False),
    ('SF binary multi-task',
     exp_path('rowsplit_binary_v2', '20260329-010643'), 'binary', True),
    ('SF 4cls cls-only',
     exp_path('rowsplit_4cls_v2_cls_only', '20260401-151749'), '4cls', False),
    ('SF 4cls multi-task',
     exp_path('rowsplit_4cls_v2', '20260327-203141'), '4cls', True),
    # Temporal models
    ('Temporal binary cls-only',
     exp_path('rowsplit_binary_temporal_v2_cls_only', '20260401-151749'), 'binary', False),
    ('Temporal binary multi-task',
     exp_path('rowsplit_binary_temporal_v2', '20260329-212856'), 'binary', True),
    ('Temporal 4cls cls-only',
     exp_path('rowsplit_4cls_temporal_v2_cls_only', '20260401-151812'), '4cls', False),
    ('Temporal 4cls multi-task',
     exp_path('rowsplit_4cls_temporal_v2', '20260327-155528'), '4cls', True),
]

TABLE2_EXPERIMENTS = [
    # offset ablation — all 4cls multi-task
    ('Single-frame [0,0,0]',
     exp_path('rowsplit_4cls_v2', '20260327-203141'), '4cls', True),
    ('Short [-3,-1.5,0]',
     exp_path('rowsplit_4cls_temporal_v2_short', '20260329-220430'), '4cls', True),
    ('Standard [-6,-3,0]',
     exp_path('rowsplit_4cls_temporal_v2', '20260327-155528'), '4cls', True),
    ('Long [-12,-6,0]',
     exp_path('rowsplit_4cls_temporal_v2_long', '20260329-220436'), '4cls', True),
]


# ---------------------------------------------------------------------------
# Run bootstrap for all experiments
# ---------------------------------------------------------------------------

def run_table(experiments, label):
    print(f'\n=== {label} ===')
    rows = []
    for row_label, path, task, has_reg in experiments:
        print(f'  {row_label} ...', end=' ', flush=True)
        samples = load_samples(path)
        ci = bootstrap_ci(samples, task=task, has_reg=has_reg)

        # Also compute point estimates from full data
        pt = compute_metrics(samples, task=task, has_reg=has_reg)

        row = {'row': row_label}
        for metric in ['accuracy', 'f1_macro', 'mae', 'r2']:
            if metric in pt:
                lo, hi = ci[metric]
                row[f'{metric}_point'] = pt[metric]
                row[f'{metric}_lo95'] = lo
                row[f'{metric}_hi95'] = hi
        rows.append(row)
        print(f"acc={pt['accuracy']:.4f} [{ci['accuracy'][0]:.4f}, {ci['accuracy'][1]:.4f}]")

    # Also compute derived binary for 4cls experiments
    print(f'\n  --- Derived binary for 4cls rows ---')
    rows_bin = []
    for row_label, path, task, has_reg in experiments:
        if task == '4cls':
            samples = load_samples(path)
            ci_bin = bootstrap_ci(samples, task='4cls_binary', has_reg=False)
            pt_bin = compute_metrics(samples, task='4cls_binary', has_reg=False)
            lo, hi = ci_bin['accuracy']
            rows_bin.append({
                'row': row_label,
                'binary_acc_point': pt_bin['accuracy'],
                'binary_acc_lo95': lo,
                'binary_acc_hi95': hi,
            })
            print(f'  {row_label}: binary={pt_bin["accuracy"]:.4f} [{lo:.4f}, {hi:.4f}]')

    return rows, rows_bin


def mcnemar_test():
    """McNemar's test: temporal 4cls MT vs SF 4cls MT, binary derived."""
    print('\n=== McNemar\'s Test: Temporal vs SF (binary derived) ===')
    path_temporal = exp_path('rowsplit_4cls_temporal_v2', '20260327-155528')
    path_sf       = exp_path('rowsplit_4cls_v2', '20260327-203141')

    temp_samples = load_samples(path_temporal)
    sf_samples   = load_samples(path_sf)

    # Sort both by index to ensure alignment
    temp_samples = sorted(temp_samples, key=lambda s: s['index'])
    sf_samples   = sorted(sf_samples,   key=lambda s: s['index'])
    assert len(temp_samples) == len(sf_samples), "Sample count mismatch"

    temp_bin = np.array([derive_binary(s['pred_label']) for s in temp_samples])
    sf_bin   = np.array([derive_binary(s['pred_label']) for s in sf_samples])
    true_bin = np.array([0 if s['true_label'] < 3 else 1 for s in temp_samples])

    # McNemar contingency: correct/incorrect for each model
    temp_correct = (temp_bin == true_bin)
    sf_correct   = (sf_bin   == true_bin)

    # b: SF correct, temporal wrong; c: temporal correct, SF wrong
    b = np.sum( sf_correct & ~temp_correct)
    c = np.sum(~sf_correct &  temp_correct)
    n_discordant = b + c

    # McNemar statistic with continuity correction
    if n_discordant == 0:
        stat, p = 0.0, 1.0
    else:
        stat = (abs(b - c) - 1.0) ** 2 / (b + c)
        # chi-squared with 1 df
        from scipy.stats import chi2
        p = 1 - chi2.cdf(stat, df=1)

    result = (
        f"McNemar's test (temporal 4cls MT vs SF 4cls MT, binary derived)\n"
        f"  b (SF correct, temporal wrong): {b}\n"
        f"  c (temporal correct, SF wrong): {c}\n"
        f"  Total discordant pairs: {n_discordant}\n"
        f"  Chi-squared statistic (continuity corrected): {stat:.4f}\n"
        f"  p-value: {p:.2e}\n"
        f"  Temporal accuracy: {temp_correct.mean():.4f}\n"
        f"  SF accuracy: {sf_correct.mean():.4f}\n"
    )
    print(result)
    return result, stat, p


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_csv(rows, path):
    if not rows:
        return
    # Collect union of all keys across rows
    all_keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction='ignore', restval='')
        w.writeheader()
        w.writerows(rows)
    print(f'  → saved {path}')


def format_ci_latex(point, lo, hi, fmt='.1%'):
    """Format as '93.5 [92.1, 94.8]' for LaTeX table."""
    return f'{point:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]'


if __name__ == '__main__':
    print('Bootstrap 95% CIs — well-level resampling, 1000 iterations')
    print(f'Using seed {SEED}')

    rows1, rows1_bin = run_table(TABLE1_EXPERIMENTS, 'Table 1')
    rows2, rows2_bin = run_table(TABLE2_EXPERIMENTS, 'Table 2')
    mcn_text, mcn_stat, mcn_p = mcnemar_test()

    save_csv(rows1, OUT_DIR / 'bootstrap_ci_table1.csv')
    save_csv(rows2, OUT_DIR / 'bootstrap_ci_table2.csv')
    if rows1_bin:
        save_csv(rows1_bin, OUT_DIR / 'bootstrap_ci_table1_binary_derived.csv')
    if rows2_bin:
        save_csv(rows2_bin, OUT_DIR / 'bootstrap_ci_table2_binary_derived.csv')

    with open(OUT_DIR / 'mcnemar_result.txt', 'w') as f:
        f.write(mcn_text)

    # Print LaTeX-ready CI strings for the main models
    print('\n=== LaTeX-ready CI strings for key rows ===')
    for row in rows1:
        if row['row'] in ('SF 4cls multi-task', 'Temporal binary multi-task',
                          'Temporal 4cls multi-task'):
            acc = row['accuracy_point']
            lo  = row['accuracy_lo95']
            hi  = row['accuracy_hi95']
            print(f"  {row['row']}: {acc:.1%} [{lo:.1%}, {hi:.1%}]")
            if 'mae_point' in row:
                print(f"    MAE: {row['mae_point']:.2f}h [{row['mae_lo95']:.2f}, {row['mae_hi95']:.2f}]")
                print(f"    R2:  {row['r2_point']:.3f} [{row['r2_lo95']:.3f}, {row['r2_hi95']:.3f}]")

    print('\nDone.')
