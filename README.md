# cell_tempo

`cell_tempo` is a PyTorch pipeline for the paper's temporal brightfield cell-state modeling experiments.

This public snapshot is organized around the main split used in the paper:

- Rows `A` and `B` are used for training.
- Row `C` is held out for testing.
- The primary training entry point is `train_rowsplit.py` with the `configs/rowsplit_*.yaml` configs.

## What is included

- Core code for datasets, models, transforms, metrics, training, and evaluation.
- Curated training configs that reflect the paper-facing experiments.
- The row-split configs and launch scripts used for the paper's main experiments and ablations.
- SLURM submission examples in `scripts/`.


## Repository layout

- `analysis/`: small paper-facing utilities retained after cleanup
- `configs/`: public configs with relative data placeholders
- `datasets/`: dataset builders and row-split logic
- `models/`: ResNet-based multitask model definition
- `scripts/`: example SLURM launch scripts
- `utils/`: config loading, logging, metrics, transforms
- `train_rowsplit.py`: main paper-aligned training entry point
- `convert_tiff_to_npy.py`: TIFF-to-NPY cache preparation

## Data paths

All public configs now use relative placeholder paths such as:

- `data/run1`
- `data/run2`
- `data/run3`

Before training, update those paths to match your local data layout or place the datasets under `data/`.

## Main workflows

### 1. Paper-aligned row split

Train with rows `A+B` and test on row `C`:

```bash
python train_rowsplit.py --config configs/rowsplit_4cls_temporal.yaml
```

Other paper-facing variants are also provided:

- `configs/rowsplit_4cls.yaml`
- `configs/rowsplit_binary_temporal.yaml`
- `configs/rowsplit_binary.yaml`
- the `*_cls_only.yaml` and `*_reg_only.yaml` ablation configs

### 2. TIFF to NPY cache conversion

```bash
python convert_tiff_to_npy.py --workers 8
```


