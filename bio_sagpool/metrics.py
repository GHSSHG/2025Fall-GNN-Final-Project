from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    loss: float
    roc_auc: float | None
    pr_auc: float | None
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "threshold": self.threshold,
        }


def best_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray, *, steps: int = 200) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thresholds = np.linspace(0.0, 1.0, steps + 1)
    f1s = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    return float(thresholds[int(np.argmax(f1s))])


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    loss: float,
    threshold: float,
) -> BinaryMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = None
    pr_auc = None
    # These metrics require both classes to be present.
    if len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    return BinaryMetrics(
        loss=float(loss),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        threshold=float(threshold),
    )

