from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn import metrics


@dataclass
class AverageMeter:
    name: str
    value: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n
        self.value = val

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def binary_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    results = {
        "accuracy": metrics.accuracy_score(labels, preds),
        "precision": metrics.precision_score(labels, preds, zero_division=0),
        "recall": metrics.recall_score(labels, preds, zero_division=0),
        "f1": metrics.f1_score(labels, preds, zero_division=0),
    }
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*ROC AUC score is not defined.*")
            results["auc"] = metrics.roc_auc_score(labels, probs)
    except (ValueError, RuntimeWarning):
        results["auc"] = float("nan")
    return results


def multiclass_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute multiclass classification metrics.

    Args:
        probs: (N, C) softmax probabilities
        labels: (N,) integer ground-truth labels
        class_names: optional list of class names for per-class reporting

    Returns:
        Dictionary with overall and per-class metrics.
    """
    n_classes = probs.shape[1]
    preds = probs.argmax(axis=1)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(n_classes)]

    results: Dict[str, float] = {}

    # Overall metrics
    results["accuracy"] = float(metrics.accuracy_score(labels, preds))

    # Macro / weighted averages
    prec_macro, rec_macro, f1_macro, _ = metrics.precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = metrics.precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    results["precision_macro"] = float(prec_macro)
    results["recall_macro"] = float(rec_macro)
    results["f1_macro"] = float(f1_macro)
    results["precision_weighted"] = float(prec_weighted)
    results["recall_weighted"] = float(rec_weighted)
    results["f1_weighted"] = float(f1_weighted)

    # Per-class metrics
    prec_per, rec_per, f1_per, sup_per = metrics.precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0, labels=list(range(n_classes))
    )
    for i, name in enumerate(class_names):
        results[f"{name}_precision"] = float(prec_per[i])
        results[f"{name}_recall"] = float(rec_per[i])
        results[f"{name}_f1"] = float(f1_per[i])
        results[f"{name}_support"] = int(sup_per[i])

    # AUC (one-vs-rest, macro)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            results["auc_macro"] = float(metrics.roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"
            ))
            results["auc_weighted"] = float(metrics.roc_auc_score(
                labels, probs, multi_class="ovr", average="weighted"
            ))
    except (ValueError, RuntimeWarning):
        results["auc_macro"] = float("nan")
        results["auc_weighted"] = float("nan")

    return results
