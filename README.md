# Cell Classification Pipeline

A PyTorch-based training and analysis pipeline for classifying live-cell imaging time-course TIFF stacks as **infected** or **uninfected**. Each TIFF contains 95 time points (t0–t94) acquired every 30 minutes over ~47 hours.

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Training](#-training)
- [Analysis Scripts](#-analysis-scripts)
- [Visualization](#-visualization)
- [Configuration](#️-configuration)
- [Documentation](#-documentation)

---

## 🚀 Quick Start

```powershell
# 1. Install dependencies
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Update data paths in configs/resnet50_baseline.yaml

# 3. Run basic training
python train.py --config configs/resnet50_baseline.yaml

# 4. Or run time window analysis
bash shells/analyze_interval_sweep_train.sh
```

---

## 📁 Project Structure

```
cell_classification/
├── configs/                          # Experiment configurations
│   ├── resnet50_baseline.yaml       # Default: 16-30h infection window
│   ├── resnet50_early.yaml          # Early detection: 0-16h window
│   └── resnet50_time_regression.yaml
│
├── datasets/
│   ├── __init__.py
│   └── timecourse_dataset.py        # TIFF dataset + CV splits
│
├── models/
│   ├── __init__.py
│   └── resnet.py                    # ResNet classifier
│
├── utils/                            # Logging, metrics, transforms
│   ├── config.py
│   ├── logger.py
│   ├── metrics.py
│   ├── seed.py
│   └── transforms.py
│
├── shells/                           # Shell scripts for experiments
│   ├── analyze_interval_sweep_train.sh
│   ├── analyze_sliding_window_train.sh
│   └── ...
│
├── docs/                             # Documentation
│   ├── INTERVAL_SWEEP_GUIDE.md
│   ├── MATCH_UNINFECTED_WINDOW.md
│   └── ...
│
├── scripts/
│   └── legacy/                       # Deprecated evaluation-only scripts
│
├── train.py                          # Main training script
├── analyze_interval_sweep_train.py  # Interval analysis (trains models)
├── analyze_sliding_window_train.py  # Window analysis (trains models)
├── visualize_cam.py                  # Grad-CAM visualization
├── test_folds.py                     # Re-evaluate checkpoints
└── requirements.txt
```

**Data Structure:**
```
DATA_ROOT/
├── infected/
│   └── *.tiff     # Multi-frame stacks (infected wells)
└── uninfected/
    └── *.tiff     # Control stacks
```

---

## 🎯 Training

### Basic Training

Standard training with default config:

```powershell
python train.py --config configs/resnet50_baseline.yaml
```

**Outputs:**
- Checkpoints: `checkpoints/<experiment>/<timestamp>/`
- Logs: `outputs/<experiment>/<timestamp>/`

### Cross-Validation

Enable K-fold cross-validation:

```powershell
python train.py --config configs/resnet50_baseline.yaml --k-folds 5
```

**Outputs:**
- Per-fold checkpoints: `checkpoints/<experiment>/<timestamp>/fold_XX of YY/best.pt`
- CV summary: `outputs/<experiment>/<timestamp>/cv_summary.json`

### Early Detection

Train on early time window (0-16h, before visible CPE):

```powershell
python train.py --config configs/resnet50_early.yaml
```

### Time Regression

Predict time-to-infection instead of binary classification:

```powershell
python train.py --config configs/resnet50_time_regression.yaml
```

---

## 📊 Analysis Scripts

### 1. Interval Sweep Training

**Purpose:** Train models with different infected interval ranges [start, x] to discover how much temporal information is needed for accurate classification.

```powershell
python analyze_interval_sweep_train.py `
    --config configs/resnet50_baseline.yaml `
    --upper-hours 7 10 13 16 19 22 25 28 31 34 37 40 43 46 `
    --start-hour 1 `
    --metrics auc accuracy f1 `
    --k-folds 5 `
    --epochs 10 `
    --mode both
```

**What it does:**
- Trains models for each interval [1, 7], [1, 10], [1, 13], etc.
- Runs two modes:
  - **test-only**: Train on full data, test on restricted intervals
  - **train-test**: Both train and test use restricted intervals
- Generates two-panel comparison plots

**Key Features:**
- ⚡ **Optimized for test-only mode**: Trains models ONCE and reuses them for all test intervals (14x faster!)
- 🎯 **--match-uninfected-window**: Apply same time window to uninfected samples for fair comparison
- 📈 **Multiple metrics**: Track AUC, accuracy, F1, precision, recall simultaneously

**Outputs:**
- `outputs/interval_sweep_analysis/<timestamp>/interval_sweep_combined.png`
- `outputs/interval_sweep_analysis/<timestamp>/interval_sweep_<metric>.png`
- `outputs/interval_sweep_analysis/<timestamp>/interval_sweep_data.json`
- `checkpoints/test-only_base_models/fold_XX_best.pth` (test-only mode models)
- `checkpoints/train-test_interval_*/fold_XX_best.pth` (train-test mode models)

**Shell scripts:**
```bash
# Unix/Linux
bash shells/analyze_interval_sweep_train.sh

# Windows PowerShell
.\shells\analyze_interval_sweep_train.ps1
```

**See also:** `docs/INTERVAL_SWEEP_GUIDE.md` for detailed usage

---

### 2. Sliding Window Training

**Purpose:** Train models on different time windows [x, x+k] to identify which periods are most predictive.

```powershell
python analyze_sliding_window_train.py `
    --config configs/resnet50_baseline.yaml `
    --window-size 5 `
    --stride 2 `
    --start-hour 0 `
    --end-hour 30 `
    --metrics auc accuracy f1 `
    --k-folds 5 `
    --epochs 10
```

**Parameters:**
- `--window-size`: Width of each window in hours (default: 5)
- `--stride`: Step between windows (default: window-size)
  - `stride < window-size`: Overlapping windows
  - `stride = window-size`: Adjacent windows
  - `stride > window-size`: Gaps between windows
- `--match-uninfected-window`: Apply same window to uninfected samples

**Example Interpretation:**
```
Window [0,5]:   AUC = 0.65 ± 0.03  ← Early period, weak signal
Window [10,15]: AUC = 0.88 ± 0.02  ← Mid-infection, strong signal
Window [20,25]: AUC = 0.95 ± 0.01  ← Late infection, very strong signal
```

**Outputs:**
- `outputs/sliding_window_analysis/<timestamp>/sliding_window_w<size>_s<stride>_combined.png`
- `outputs/sliding_window_analysis/<timestamp>/sliding_window_w<size>_s<stride>_<metric>.png`
- `outputs/sliding_window_analysis/<timestamp>/sliding_window_w<size>_s<stride>_data.json`
- `checkpoints/window_*/fold_XX_best.pth` (trained models for each window)

**Shell scripts:**
```bash
# Unix/Linux
bash shells/analyze_sliding_window_train.sh

# Windows PowerShell
.\shells\analyze_sliding_window_train.ps1
```

---

### 3. Match Uninfected Window Feature

**NEW:** Both analysis scripts now support `--match-uninfected-window` flag to apply the same time interval to both infected and uninfected samples for fair comparison.

**Without flag (default):**
- Infected: restricted to [x, x+k]
- Uninfected: uses ALL time points

**With flag:**
- Infected: restricted to [x, x+k]
- Uninfected: restricted to [x, x+k] ← **Same window!**

**Usage:**
```powershell
# Enable in Python
python analyze_interval_sweep_train.py --config ... --match-uninfected-window

# Or set MATCH_UNINFECTED=true in shell scripts
```

**See also:** `docs/MATCH_UNINFECTED_WINDOW.md` and `BUGFIX_UNINFECTED_WINDOW.md`

---

## 🔬 Visualization

### Grad-CAM Heatmaps

Visualize which regions the model focuses on:

```powershell
python visualize_cam.py `
    --config configs/resnet50_baseline.yaml `
    --checkpoint checkpoints/resnet50_baseline/20251204-141200/fold_01of05/best.pt `
    --split val `
    --num-samples 8
```

**Outputs:** `cam_outputs/<checkpoint>/<split>/`
- Raw images
- Heatmaps
- Overlays with predictions
- `metadata.json`

### Re-Test Checkpoints

Re-evaluate all saved checkpoints from a training run:

```powershell
python test_folds.py `
    --config configs/resnet50_baseline.yaml `
    --run-dir checkpoints/resnet50_baseline/20251205-101010 `
    --split test
```

---

## ⚙️ Configuration

### Key Config Sections

**Data Configuration:**
```yaml
data:
  infected_dir: "/path/to/infected/"
  uninfected_dir: "/path/to/uninfected/"
  batch_size: 256
  eval_batch_size_multiplier: 2    # Eval uses 512 (256*2) for speed
  num_workers: 4
  balance_sampler: false            # Use weighted sampler for class balance
  
  frames:
    frames_per_hour: 2.0
    infected_window_hours: [16, 30]  # CPE window
    infected_stride: 1
    uninfected_use_all: true
    
    # Optional: Split-specific overrides
    test:
      infected_window_hours: [1, 12]  # Test early detection only
```

**Model Configuration:**
```yaml
model:
  name: resnet50                     # resnet18/34/50/101/152
  pretrained: true
```

**Training Configuration:**
```yaml
training:
  epochs: 30
  k_folds: 1                         # Set >1 for cross-validation
  amp: true                          # Automatic mixed precision

optimizer:
  lr: 1e-4
  weight_decay: 1e-5

scheduler:
  type: cosine
  t_max: 30
  eta_min: 1e-6
```

**Task Configuration:**
```yaml
task:
  type: classification               # or "regression"
```

**Analysis Configuration:**
```yaml
analysis:
  thresholds: [0.1, 0.3, 0.5, 0.7, 0.9]  # Precision/recall at each threshold
  time_bins:                              # Metrics per time window
    - [0, 6]
    - [6, 12]
    - [12, 18]
    - [18, 24]
```

### Performance Tuning

**Batch Sizes:**
- Training uses `batch_size` (e.g., 256)
- Eval uses `batch_size * eval_batch_size_multiplier` (e.g., 512)
- Eval can be 2-3x larger since no gradients are computed

**Recommended multipliers:**
- 16+ GB GPU: `eval_batch_size_multiplier: 3`
- 8-16 GB GPU: `eval_batch_size_multiplier: 2` (default)
- <8 GB GPU: `eval_batch_size_multiplier: 1`

---

## 📚 Documentation

Detailed guides in the `docs/` folder:

- **[INTERVAL_SWEEP_GUIDE.md](docs/INTERVAL_SWEEP_GUIDE.md)** - Comprehensive interval sweep guide
- **[MATCH_UNINFECTED_WINDOW.md](docs/MATCH_UNINFECTED_WINDOW.md)** - Uninfected window matching feature
- **[EXPERIMENTS_README.md](docs/EXPERIMENTS_README.md)** - Experiment organization
- **[MODE_CLARIFICATION.md](docs/MODE_CLARIFICATION.md)** - train-test vs test-only modes
- **[ANALYSIS_UPDATES.md](docs/ANALYSIS_UPDATES.md)** - Analysis script updates

Recent updates:

- **[OPTIMIZATION_TEST_ONLY.md](OPTIMIZATION_TEST_ONLY.md)** - Test-only mode optimization (14x speedup!)
- **[BUGFIX_UNINFECTED_WINDOW.md](BUGFIX_UNINFECTED_WINDOW.md)** - Critical bug fix for window matching
- **[SHELL_SCRIPTS_UPDATE.md](SHELL_SCRIPTS_UPDATE.md)** - Shell script modifications

---

## 🔧 Advanced Features

### Frame Policy

Control which frames are sampled from each TIFF:

- **infected_window_hours**: Time range for infected samples (default: [16, 30])
- **uninfected_use_all**: Use all frames from uninfected samples (default: true)
- **infected_stride**: Subsample infected frames (default: 1 = all frames)
- **uninfected_stride**: Subsample uninfected frames (default: 1)

### Split-Specific Policies

Override frame policies per split:

```yaml
data:
  frames:
    infected_window_hours: [1, 30]     # Default for all splits
    
    # Override for specific splits
    train:
      infected_window_hours: [1, 30]   # Train on full range
    test:
      infected_window_hours: [1, 12]   # Test early detection
```

### Threshold Tuning

When `analysis.thresholds` is set, the script logs precision/recall/F1 for each threshold:

```
thr=0.10 | P=0.23 | R=0.95 | F1=0.38 | TP=74
thr=0.20 | P=1.00 | R=0.03 | F1=0.05 | TP=2
thr=0.50 | P=0.00 | R=0.00 | F1=0.00 | TP=0
```

Use this to select an operating point (e.g., recall ≥ 0.7).

### Time Bin Analysis

`analysis.time_bins` computes full metrics inside every bin:

```
val time bin [0.0h, 8.0h): samples=174 | accuracy:0.552 | auc:0.741
val time bin [8.0h, 12.0h): samples=48 | accuracy:1.000 | auc:nan
```

---

## 📝 Notes

- **TIFF compression**: Requires `imagecodecs` package (included in `requirements.txt`)
- **PyTorch 2.6+**: Use `--weights-only` flag for safer checkpoint loading
- **Frame expansion**: Each TIFF is expanded into multiple samples based on frame policy
- **Checkpoints**: Automatically save best model per fold based on validation metric
- **Mixed precision**: Enabled by default (`amp: true`) for faster training

---

## 🐛 Recent Bug Fixes

### Critical: Uninfected Window Matching (Dec 12, 2025)

**Issue:** The `--match-uninfected-window` flag was being silently ignored. Uninfected samples always used ALL time points even when the flag was enabled.

**Fix:** Added `uninfected_window_hours` field to `FrameExtractionPolicy` class in `datasets/timecourse_dataset.py`.

**Impact:** ALL experiments with `--match-uninfected-window` before this fix are INVALID and should be re-run.

See `BUGFIX_UNINFECTED_WINDOW.md` for details.

### Optimization: Test-Only Mode (Dec 12, 2025)

**Before:** Test-only mode trained 70 models (14 intervals × 5 folds)  
**After:** Test-only mode trains 5 models (1 per fold), reused for all 14 intervals  
**Speedup:** 14x faster! (46% reduction in total training runs)

See `OPTIMIZATION_TEST_ONLY.md` for details.

---

## 🎯 Quick Examples

**Train with 5-fold CV:**
```powershell
python train.py --config configs/resnet50_baseline.yaml --k-folds 5
```

**Find best time window:**
```bash
bash shells/analyze_sliding_window_train.sh
```

**Compare test-only vs train-test:**
```bash
bash shells/run_both_experiments.sh
```

**Visualize model decisions:**
```powershell
python visualize_cam.py --checkpoint checkpoints/.../best.pt --split val
```

**Fair infected/uninfected comparison:**
```powershell
python analyze_interval_sweep_train.py --config ... --match-uninfected-window
```

---

## 📧 Support

For questions or issues:
1. Check `docs/` folder for detailed guides
2. Review recent documentation updates in root folder
3. Check shell scripts for example usage

---

## 📜 License

[Your License Here]
