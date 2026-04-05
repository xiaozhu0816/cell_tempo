"""
5-Fold Cross-Validation Training for Multitask Model

This script runs stratified 5-fold cross-validation with the same settings
as the standard training, enabling robust performance evaluation and
fair comparison with single-task models.

Usage:
    python train_multitask_cv.py --config configs/multitask_example.yaml --num-folds 5
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from local modules
from datasets import build_datasets, format_policy_summary, resolve_frame_policies
from models import build_multitask_model
from utils import AverageMeter, binary_metrics, build_transforms, get_logger, load_config, set_seed


def meta_batch_to_list(meta_batch) -> List[Dict[str, Any]]:
    """Convert batched metadata to list of dicts."""
    if isinstance(meta_batch, list):
        return meta_batch
    if isinstance(meta_batch, dict):
        keys = list(meta_batch.keys())
        if not keys:
            return []
        length = len(meta_batch[keys[0]])
        meta_list: List[Dict[str, Any]] = []
        for i in range(length):
            entry: Dict[str, Any] = {}
            for key in keys:
                value = meta_batch[key][i]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                entry[key] = value
            meta_list.append(entry)
        return meta_list
    raise TypeError(f"Unsupported meta batch type: {type(meta_batch)}")


def build_multitask_targets(
    labels: torch.Tensor,
    meta_list: List[Dict[str, Any]],
    infection_onset_hour: float,
    clamp_range: Tuple[float, float],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build targets for multi-task learning."""
    clamp_min, clamp_max = clamp_range
    
    cls_targets = labels.long().to(device, non_blocking=True)
    
    time_list: List[float] = []
    for label, meta in zip(labels.tolist(), meta_list):
        hours = float(meta.get("hours_since_start", 0.0))
        
        if int(label) == 1:  # Infected
            time_value = max(hours - infection_onset_hour, 0.0)
        else:  # Uninfected
            time_value = hours
        
        if clamp_min is not None:
            time_value = max(time_value, clamp_min)
        if clamp_max is not None:
            time_value = min(time_value, clamp_max)
        
        time_list.append(time_value)
    
    time_targets = torch.tensor(time_list, dtype=torch.float32).unsqueeze(1)
    return cls_targets, time_targets.to(device, non_blocking=True)


def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    diff = preds - targets
    mae = np.abs(diff).mean()
    mse = (diff ** 2).mean()
    rmse = np.sqrt(mse)
    return {"mae": mae, "rmse": rmse, "mse": mse}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    cls_criterion: nn.Module,
    reg_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    infection_onset_hour: float,
    clamp_range: Tuple[float, float],
    cls_weight: float = 1.0,
    reg_weight: float = 1.0,
    use_amp: bool = True,
    grad_clip: Optional[float] = None,
    progress_desc: str = "train",
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss_meter = AverageMeter("total_loss")
    cls_loss_meter = AverageMeter("cls_loss")
    reg_loss_meter = AverageMeter("reg_loss")
    
    for images, labels, meta in tqdm(loader, desc=progress_desc, leave=False):
        images = images.to(device, non_blocking=True)
        meta_list = meta_batch_to_list(meta)
        
        cls_targets, time_targets = build_multitask_targets(
            labels, meta_list, infection_onset_hour, clamp_range, device
        )
        
        optimizer.zero_grad()
        
        # Use autocast context (PyTorch version compatible)
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                cls_logits, time_pred = model(images)
                cls_loss = cls_criterion(cls_logits, cls_targets)
                reg_loss = reg_criterion(time_pred, time_targets)
                total_loss = cls_weight * cls_loss + reg_weight * reg_loss
        else:
            cls_logits, time_pred = model(images)
            cls_loss = cls_criterion(cls_logits, cls_targets)
            reg_loss = reg_criterion(time_pred, time_targets)
            total_loss = cls_weight * cls_loss + reg_weight * reg_loss
        
        scaler.scale(total_loss).backward()
        
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        batch_size = images.size(0)
        total_loss_meter.update(total_loss.item(), n=batch_size)
        cls_loss_meter.update(cls_loss.item(), n=batch_size)
        reg_loss_meter.update(reg_loss.item(), n=batch_size)
    
    return {
        "total_loss": total_loss_meter.avg,
        "cls_loss": cls_loss_meter.avg,
        "reg_loss": reg_loss_meter.avg,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cls_criterion: nn.Module,
    reg_criterion: nn.Module,
    device: torch.device,
    infection_onset_hour: float,
    clamp_range: Tuple[float, float],
    cls_weight: float = 1.0,
    reg_weight: float = 1.0,
    split_name: str = "val",
    progress_desc: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Evaluate model."""
    model.eval()
    
    total_loss_meter = AverageMeter(f"{split_name}_total_loss")
    cls_loss_meter = AverageMeter(f"{split_name}_cls_loss")
    reg_loss_meter = AverageMeter(f"{split_name}_reg_loss")
    
    cls_logits_list, cls_targets_list = [], []
    time_preds_list, time_targets_list = [], []
    
    desc = progress_desc or split_name
    
    with torch.no_grad():
        for images, labels, meta in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            meta_list = meta_batch_to_list(meta)
            
            cls_targets, time_targets = build_multitask_targets(
                labels, meta_list, infection_onset_hour, clamp_range, device
            )
            
            cls_logits, time_pred = model(images)
            
            cls_loss = cls_criterion(cls_logits, cls_targets)
            reg_loss = reg_criterion(time_pred, time_targets)
            total_loss = cls_weight * cls_loss + reg_weight * reg_loss
            
            batch_size = images.size(0)
            total_loss_meter.update(total_loss.item(), n=batch_size)
            cls_loss_meter.update(cls_loss.item(), n=batch_size)
            reg_loss_meter.update(reg_loss.item(), n=batch_size)
            
            cls_logits_list.append(cls_logits.detach().cpu().numpy())
            cls_targets_list.append(cls_targets.cpu().numpy())
            time_preds_list.append(time_pred.detach().cpu().numpy())
            time_targets_list.append(time_targets.cpu().numpy())
    
    if not cls_logits_list:
        return {"total_loss": total_loss_meter.avg}, {}
    
    all_cls_logits = np.concatenate(cls_logits_list, axis=0)
    all_cls_targets = np.concatenate(cls_targets_list, axis=0)
    
    from scipy.special import softmax
    all_cls_probs = softmax(all_cls_logits, axis=1)[:, 1]
    
    epsilon = 1e-7
    all_cls_probs = np.clip(all_cls_probs, epsilon, 1 - epsilon)
    all_cls_logits_binary = np.log(all_cls_probs / (1 - all_cls_probs))
    
    cls_metrics = binary_metrics(all_cls_logits_binary, all_cls_targets)
    
    all_time_preds = np.concatenate(time_preds_list, axis=0).squeeze(-1)
    all_time_targets = np.concatenate(time_targets_list, axis=0).squeeze(-1)
    reg_metrics = compute_regression_metrics(all_time_preds, all_time_targets)
    
    metrics = {
        "total_loss": total_loss_meter.avg,
        "cls_loss": cls_loss_meter.avg,
        "reg_loss": reg_loss_meter.avg,
    }
    
    for k, v in cls_metrics.items():
        metrics[f"cls_{k}"] = v
    
    for k, v in reg_metrics.items():
        metrics[f"reg_{k}"] = v
    
    # Compute combined metric
    cls_f1 = metrics.get("cls_f1", 0.0)
    reg_mae = metrics.get("reg_mae", clamp_range[1])
    max_time = clamp_range[1]
    
    reg_score = max(0.0, 1.0 - (reg_mae / max_time))
    combined_metric = 0.6 * cls_f1 + 0.4 * reg_score
    metrics["combined"] = combined_metric
    
    predictions = {
        "time_preds": all_time_preds,
        "time_targets": all_time_targets,
        "cls_preds": all_cls_probs,
        "cls_targets": all_cls_targets,
    }
    
    return metrics, predictions


def train_single_fold(
    fold_idx: int,
    num_folds: int,
    cfg: Dict,
    output_base: Path,
    logger,
) -> Dict[str, Any]:
    """Train a single fold."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"FOLD {fold_idx + 1}/{num_folds}")
    logger.info(f"{'='*80}\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Multitask config
    mt_cfg = cfg.get("multitask", {})
    infection_onset_hour = float(mt_cfg.get("infection_onset_hour", 2.0))
    clamp_range = tuple(mt_cfg.get("clamp_range", [0.0, 48.0]))
    cls_weight = float(mt_cfg.get("classification_weight", 1.0))
    reg_weight = float(mt_cfg.get("regression_weight", 1.0))
    
    # Build datasets for this fold
    data_cfg = cfg.get("data", {})
    transforms_dict = build_transforms(data_cfg.get("transforms", {}))
    
    logger.info(f"Building datasets for fold {fold_idx + 1}...")
    train_ds, val_ds, test_ds = build_datasets(
        data_cfg=data_cfg,
        transforms=transforms_dict,
        fold_index=fold_idx,
        num_folds=num_folds,
    )
    
    logger.info(f"Fold {fold_idx + 1} - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Data loaders
    batch_size = data_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Build model
    model_cfg = cfg.get("model", {})
    model = build_multitask_model(model_cfg)
    model = model.to(device)
    
    # Optimizer
    opt_cfg = cfg.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get("lr", 1e-4),
        weight_decay=opt_cfg.get("weight_decay", 1e-4),
    )
    
    # Scheduler
    scheduler_cfg = cfg.get("scheduler", {})
    training_cfg = cfg.get("training", {})
    epochs = training_cfg.get("epochs", 10)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_cfg.get("t_max", epochs),
        eta_min=scheduler_cfg.get("eta_min", 1e-6),
    )
    
    # Loss functions
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss()
    
    # Training setup
    use_amp = training_cfg.get("amp", True)
    grad_clip = training_cfg.get("grad_clip")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    
    # Checkpointing
    fold_checkpoint_dir = output_base / f"fold_{fold_idx + 1}" / "checkpoints"
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = fold_checkpoint_dir / "best.pt"
    
    primary_metric = "combined"
    best_score = -math.inf
    
    logger.info(f"Training fold {fold_idx + 1} for {epochs} epochs...")
    logger.info(f"Primary metric: {primary_metric} (0.6*F1 + 0.4*(1-MAE/{clamp_range[1]}))")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        logger.info(f"Fold {fold_idx + 1}, Epoch {epoch}/{epochs}")
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            cls_criterion=cls_criterion,
            reg_criterion=reg_criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            infection_onset_hour=infection_onset_hour,
            clamp_range=clamp_range,
            cls_weight=cls_weight,
            reg_weight=reg_weight,
            use_amp=use_amp,
            grad_clip=grad_clip,
            progress_desc=f"F{fold_idx+1}_E{epoch}_train",
        )
        
        # Validate
        val_metrics, _ = evaluate(
            model=model,
            loader=val_loader,
            cls_criterion=cls_criterion,
            reg_criterion=reg_criterion,
            device=device,
            infection_onset_hour=infection_onset_hour,
            clamp_range=clamp_range,
            cls_weight=cls_weight,
            reg_weight=reg_weight,
            split_name="val",
            progress_desc=f"F{fold_idx+1}_E{epoch}_val",
        )
        
        scheduler.step()
        
        # Save best model
        metric_value = val_metrics.get(primary_metric)
        if metric_value is not None and not math.isnan(metric_value):
            if metric_value > best_score:
                best_score = metric_value
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "config": cfg,
                        "fold_index": fold_idx,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    best_checkpoint,
                )
                logger.info(f"✓ Fold {fold_idx + 1}: New best model! {primary_metric}={best_score:.4f}")
    
    # Load best model for final test
    logger.info(f"Loading best model for fold {fold_idx + 1}...")
    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    
    # Final test evaluation
    logger.info(f"Fold {fold_idx + 1}: Final evaluation on test set")
    test_metrics, test_predictions = evaluate(
        model=model,
        loader=test_loader,
        cls_criterion=cls_criterion,
        reg_criterion=reg_criterion,
        device=device,
        infection_onset_hour=infection_onset_hour,
        clamp_range=clamp_range,
        cls_weight=cls_weight,
        reg_weight=reg_weight,
        split_name="test",
        progress_desc=f"F{fold_idx+1}_test",
    )

    # Save per-sample predictions for downstream analysis
    # Many analysis scripts expect: <result-dir>/fold_*/test_predictions.npz
    fold_dir = output_base / f"fold_{fold_idx + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = fold_dir / "test_predictions.npz"

    try:
        np.savez(
            predictions_file,
            time_preds=test_predictions.get("time_preds"),
            time_targets=test_predictions.get("time_targets"),
            cls_preds=test_predictions.get("cls_preds"),
            cls_targets=test_predictions.get("cls_targets"),
        )
        logger.info(f"✓ Saved test predictions to {predictions_file}")
    except Exception as e:
        logger.warning(f"Failed to save {predictions_file}: {e}")

    # Also save lightweight per-sample metadata (time/position/tif/etc) if available.
    # This helps analysis scripts that stratify by time/position without reloading the dataset.
    metadata_file = fold_dir / "test_metadata.jsonl"
    try:
        if hasattr(test_ds, "get_metadata"):
            with open(metadata_file, "w", encoding="utf-8") as f:
                for i in range(len(test_ds)):
                    meta = test_ds.get_metadata(i)
                    if not isinstance(meta, dict):
                        meta = {"meta": meta}
                    f.write(json.dumps(meta) + "\n")
            logger.info(f"✓ Saved test metadata to {metadata_file}")
    except Exception as e:
        logger.warning(f"Failed to save {metadata_file}: {e}")
    
    # Log test results
    summary = " | ".join(f"{k}:{v:.4f}" for k, v in test_metrics.items())
    logger.info(f"Fold {fold_idx + 1} test results: {summary}")
    
    # Save fold results
    fold_results = {
        "fold_index": fold_idx,
        "best_val_metric": float(best_score),
        "test_metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in test_metrics.items()},
    }
    
    fold_results_file = fold_dir / "results.json"
    with open(fold_results_file, "w") as f:
        json.dump(fold_results, f, indent=2)
    
    logger.info(f"✓ Fold {fold_idx + 1} complete!")
    
    return fold_results


def aggregate_cv_results(fold_results: List[Dict], output_base: Path, logger):
    """Aggregate results across all folds."""
    
    logger.info(f"\n{'='*80}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*80}\n")
    
    # Collect metrics across folds
    metrics_names = list(fold_results[0]["test_metrics"].keys())
    
    aggregated = {}
    for metric_name in metrics_names:
        values = [fold["test_metrics"][metric_name] for fold in fold_results]
        aggregated[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": [float(v) for v in values],
        }
    
    # Log summary
    logger.info("Mean ± Std across folds:")
    for metric_name, stats in aggregated.items():
        logger.info(f"  {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Save aggregated results
    cv_summary = {
        "num_folds": len(fold_results),
        "fold_results": fold_results,
        "aggregated_metrics": aggregated,
    }
    
    summary_file = output_base / "cv_summary.json"
    with open(summary_file, "w") as f:
        json.dump(cv_summary, f, indent=2)
    
    logger.info(f"\n✓ CV summary saved to {summary_file}")
    
    return cv_summary


def evaluate_temporal_generalization_fold(
    model,
    test_loader,
    device: torch.device,
    window_size: float = 6.0,
    stride: float = 3.0,
    max_time: float = 48.0,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Evaluate temporal generalization for a single fold."""
    model.eval()
    
    # Collect all predictions
    all_cls_logits = []
    all_hours = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, meta in tqdm(test_loader, desc="Temporal analysis", leave=False):
            images = images.to(device, non_blocking=True)
            cls_logits, _ = model(images)
            
            all_cls_logits.append(cls_logits.cpu().numpy())
            all_labels.append(labels.numpy())
            
            # Extract hours from metadata
            if isinstance(meta, dict):
                hours = meta.get("hours_since_start")
                if isinstance(hours, torch.Tensor):
                    hours = hours.numpy()
                all_hours.append(hours)
            elif isinstance(meta, list):
                hours = np.array([m.get("hours_since_start", 0.0) for m in meta])
                all_hours.append(hours)
    
    # Concatenate
    cls_logits = np.concatenate(all_cls_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    hours = np.concatenate(all_hours, axis=0)
    
    # Convert logits to probabilities
    cls_probs = softmax(cls_logits, axis=1)[:, 1]
    
    # Sliding window analysis
    window_centers = []
    metrics_by_window = {
        "auc": [],
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }
    
    start = 0.0
    while start + window_size <= max_time:
        end = start + window_size
        center = (start + end) / 2.0
        
        # Get samples in this window
        mask = (hours >= start) & (hours < end)
        n_samples = mask.sum()
        
        if n_samples < 10:  # Skip windows with too few samples
            start += stride
            continue
        
        window_probs = cls_probs[mask]
        window_labels = labels[mask]
        window_pred_binary = (window_probs >= 0.5).astype(int)
        
        unique_labels = np.unique(window_labels)
        
        try:
            # AUC (need both classes)
            if len(unique_labels) > 1:
                auc = roc_auc_score(window_labels, window_probs)
            else:
                auc = None
            
            # Other metrics
            acc = accuracy_score(window_labels, window_pred_binary)
            prec, rec, f1, _ = precision_recall_fscore_support(
                window_labels, window_pred_binary, average="binary", zero_division=0
            )
            
            window_centers.append(center)
            metrics_by_window["auc"].append(auc)
            metrics_by_window["accuracy"].append(acc)
            metrics_by_window["f1"].append(f1)
            metrics_by_window["precision"].append(prec)
            metrics_by_window["recall"].append(rec)
        
        except Exception:
            pass
        
        start += stride
    
    return window_centers, metrics_by_window


def aggregate_cv_temporal_results(
    fold_results: List[Tuple[List[float], Dict[str, List[float]]]],
) -> Tuple[List[float], Dict[str, Dict[str, List[float]]]]:
    """Aggregate temporal results across folds (mean ± std)."""
    
    # Assume all folds have same window centers
    window_centers = fold_results[0][0]
    n_windows = len(window_centers)
    
    # Collect all metrics across folds
    aggregated = {}
    
    metric_names = fold_results[0][1].keys()
    
    for metric in metric_names:
        values_per_window = [[] for _ in range(n_windows)]
        
        for fold_centers, fold_metrics in fold_results:
            for i, value in enumerate(fold_metrics[metric]):
                if value is not None:
                    values_per_window[i].append(value)
        
        # Compute mean and std
        means = []
        stds = []
        
        for values in values_per_window:
            if values:
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                means.append(None)
                stds.append(None)
        
        aggregated[metric] = {
            "mean": means,
            "std": stds,
        }
    
    return window_centers, aggregated


def plot_cv_temporal_generalization(
    window_centers: List[float],
    aggregated_metrics: Dict[str, Dict[str, List[float]]],
    window_size: float,
    output_path: Path,
    num_folds: int = 5,
) -> None:
    """Create temporal generalization plot with mean ± std across folds."""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    markers = ['o', 's', '^', 'D', 'v']
    
    metric_labels = {
        "auc": "AUC",
        "accuracy": "Accuracy",
        "f1": "F1 Score",
        "precision": "Precision",
        "recall": "Recall",
    }
    
    for idx, (metric, stats) in enumerate(aggregated_metrics.items()):
        means = stats["mean"]
        stds = stats["std"]
        
        # Filter out None values
        valid_points = [
            (c, m, s) for c, m, s in zip(window_centers, means, stds)
            if m is not None and s is not None
        ]
        
        if not valid_points:
            continue
        
        centers, mean_vals, std_vals = zip(*valid_points)
        centers = np.array(centers)
        mean_vals = np.array(mean_vals)
        std_vals = np.array(std_vals)
        
        # Plot mean line
        ax.plot(
            centers,
            mean_vals,
            marker=markers[idx],
            color=colors[idx],
            label=metric_labels[metric],
            linewidth=2.5,
            markersize=8,
            alpha=0.85,
        )
        
        # Add ± std shaded region
        ax.fill_between(
            centers,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=colors[idx],
            alpha=0.2,
        )
    
    ax.set_xlabel("Time Window Center (hours)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Metric Value", fontsize=14, fontweight='bold')
    ax.set_title(
        f"Multitask Model - Temporal Generalization ({num_folds}-Fold CV)\n"
        f"Mean ± Std (Window Size: {window_size}h, Sliding Analysis)",
        fontsize=16,
        fontweight='bold',
        pad=20,
    )
    
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Auto-scale y-axis starting from minimum value (with small padding)
    all_values = []
    for stats in aggregated_metrics.values():
        all_values.extend([v for v in stats["mean"] if v is not None])
    
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        padding = y_range * 0.1 if y_range > 0 else 0.05
        ax.set_ylim(max(0, y_min - padding), min(1.05, y_max + padding))
    else:
        ax.set_ylim(0.0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved CV temporal generalization plot to {output_path}")
    plt.close()


def generate_cv_temporal_analysis(
    fold_results_list: List[Dict],
    cfg: Dict,
    output_base: Path,
    logger,
) -> None:
    """Generate temporal generalization analysis after CV training completes."""
    
    logger.info("\n" + "="*80)
    logger.info("GENERATING TEMPORAL GENERALIZATION ANALYSIS")
    logger.info("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_folds = len(fold_results_list)
    
    # Build transforms and datasets
    data_cfg = cfg.get("data", {})
    transforms_dict = build_transforms(data_cfg.get("transforms", {}))
    
    # Process each fold
    temporal_fold_results = []
    
    for fold_idx in range(num_folds):
        fold_dir = output_base / f"fold_{fold_idx + 1}"
        checkpoint_file = fold_dir / "checkpoints" / "best.pt"
        
        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint not found for fold {fold_idx + 1}, skipping")
            continue
        
        logger.info(f"Processing fold {fold_idx + 1}/{num_folds}...")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        
        # Build model
        model_cfg = cfg.get("model", {})
        model = build_multitask_model(model_cfg)
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device)
        model.eval()
        
        # Build test dataset for this fold
        _, _, test_ds = build_datasets(
            data_cfg=data_cfg,
            transforms=transforms_dict,
            fold_index=fold_idx,
            num_folds=num_folds,
        )
        
        # Create test loader
        batch_size = data_cfg.get("batch_size", 32)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=True,
        )
        
        # Evaluate temporal generalization
        window_centers, metrics_by_window = evaluate_temporal_generalization_fold(
            model=model,
            test_loader=test_loader,
            device=device,
            window_size=6.0,
            stride=3.0,
            max_time=48.0,
        )
        
        temporal_fold_results.append((window_centers, metrics_by_window))
        
        # Save individual fold results
        fold_temporal_file = fold_dir / "temporal_metrics.json"
        with open(fold_temporal_file, "w") as f:
            json.dump({
                "window_centers": window_centers,
                "metrics": {k: v for k, v in metrics_by_window.items()},
                "window_size": 6.0,
                "stride": 3.0,
            }, f, indent=2)
        logger.info(f"  ✓ Saved fold {fold_idx + 1} temporal metrics")
    
    if not temporal_fold_results:
        logger.warning("No temporal results collected")
        return
    
    # Aggregate across folds
    logger.info(f"Aggregating temporal results across {len(temporal_fold_results)} folds...")
    window_centers, aggregated_metrics = aggregate_cv_temporal_results(temporal_fold_results)
    
    # Save aggregated results
    cv_temporal_file = output_base / "cv_temporal_metrics.json"
    with open(cv_temporal_file, "w") as f:
        json.dump({
            "num_folds": len(temporal_fold_results),
            "window_centers": window_centers,
            "aggregated_metrics": {
                metric: {
                    "mean": [float(v) if v is not None else None for v in stats["mean"]],
                    "std": [float(v) if v is not None else None for v in stats["std"]],
                }
                for metric, stats in aggregated_metrics.items()
            },
            "window_size": 6.0,
            "stride": 3.0,
        }, f, indent=2)
    logger.info(f"✓ Saved CV temporal metrics to {cv_temporal_file}")
    
    # Plot aggregated results
    logger.info("Generating CV temporal generalization plot...")
    plot_path = output_base / "cv_temporal_generalization.png"
    plot_cv_temporal_generalization(
        window_centers=window_centers,
        aggregated_metrics=aggregated_metrics,
        window_size=6.0,
        output_path=plot_path,
        num_folds=len(temporal_fold_results),
    )
    
    logger.info("✓ Temporal generalization analysis complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--num-folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: outputs/multitask_cv)")
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Setup output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        experiment_name = cfg.get("experiment_name", "multitask_cv")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_base = Path("outputs") / experiment_name / f"{timestamp}_{args.num_folds}fold"
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = get_logger("multitask_cv", output_base / "train_cv.log")
    
    logger.info("="*80)
    logger.info("MULTI-TASK MODEL - CROSS-VALIDATION TRAINING")
    logger.info("="*80)
    logger.info(f"Number of folds: {args.num_folds}")
    logger.info(f"Output directory: {output_base}")
    
    # Set seed
    seed = cfg.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Train each fold
    fold_results = []
    for fold_idx in range(args.num_folds):
        fold_result = train_single_fold(
            fold_idx=fold_idx,
            num_folds=args.num_folds,
            cfg=cfg,
            output_base=output_base,
            logger=logger,
        )
        fold_results.append(fold_result)
    
    # Aggregate results
    cv_summary = aggregate_cv_results(fold_results, output_base, logger)
    
    # Generate temporal generalization analysis
    generate_cv_temporal_analysis(fold_results, cfg, output_base, logger)
    
    logger.info("\n" + "="*80)
    logger.info("CROSS-VALIDATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_base}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
