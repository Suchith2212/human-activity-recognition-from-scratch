"""
Evaluation metrics implemented from scratch using only NumPy.

Demonstrates understanding of how classification metrics work under the hood,
rather than relying solely on scikit-learn.  All functions follow the
``(y_true, y_pred)`` parameter ordering convention.

Implemented Metrics:
    - Accuracy
    - Precision (per-class and macro-averaged)
    - Recall (per-class and macro-averaged)
    - F1-Score (per-class and macro-averaged)
    - Confusion Matrix
    - Classification Report (formatted string)
"""

from typing import List, Optional, Union

import numpy as np


def _check_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true shape {y_true.shape} != y_pred shape {y_pred.shape}"
        )


# ============================================================================
# Core Metrics
# ============================================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels of shape ``(n_samples,)``.
    y_pred : np.ndarray
        Predicted labels of shape ``(n_samples,)``.

    Returns
    -------
    float
        Fraction of correctly classified samples in ``[0, 1]``.

    Examples
    --------
    >>> accuracy(np.array([1, 2, 3]), np.array([1, 2, 2]))
    0.6666666666666666
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    _check_shapes(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cls: Optional[int] = None,
    average: Optional[str] = None,
) -> Union[float, np.ndarray]:
    """Compute precision: TP / (TP + FP).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    cls : int, optional
        Specific class to compute precision for.  If ``None`` and
        ``average`` is specified, computes for all classes.
    average : {"macro", "weighted"}, optional
        Averaging strategy for multi-class precision.

    Returns
    -------
    float or np.ndarray
        Precision value(s).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    _check_shapes(y_true, y_pred)

    if cls is not None:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions = np.array([precision(y_true, y_pred, c) for c in classes])

    if average == "macro":
        return float(np.mean(precisions))
    elif average == "weighted":
        weights = np.array([np.sum(y_true == c) for c in classes])
        if np.sum(weights) == 0:
            return 0.0
        return float(np.average(precisions, weights=weights))
    return precisions


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cls: Optional[int] = None,
    average: Optional[str] = None,
) -> Union[float, np.ndarray]:
    """Compute recall: TP / (TP + FN).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    cls : int, optional
        Specific class to compute recall for.
    average : {"macro", "weighted"}, optional
        Averaging strategy for multi-class recall.

    Returns
    -------
    float or np.ndarray
        Recall value(s).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    _check_shapes(y_true, y_pred)

    if cls is not None:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    classes = np.unique(np.concatenate([y_true, y_pred]))
    recalls = np.array([recall(y_true, y_pred, c) for c in classes])

    if average == "macro":
        return float(np.mean(recalls))
    elif average == "weighted":
        weights = np.array([np.sum(y_true == c) for c in classes])
        if np.sum(weights) == 0:
            return 0.0
        return float(np.average(recalls, weights=weights))
    return recalls


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cls: Optional[int] = None,
    average: Optional[str] = None,
) -> Union[float, np.ndarray]:
    """Compute F1-score: harmonic mean of precision and recall.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    cls : int, optional
        Specific class to compute F1 for.
    average : {"macro", "weighted"}, optional
        Averaging strategy for multi-class F1.

    Returns
    -------
    float or np.ndarray
        F1 value(s).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    _check_shapes(y_true, y_pred)

    if cls is not None:
        p = precision(y_true, y_pred, cls)
        r = recall(y_true, y_pred, cls)
        return float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = np.array([f1_score(y_true, y_pred, c) for c in classes])

    if average == "macro":
        return float(np.mean(f1s))
    elif average == "weighted":
        weights = np.array([np.sum(y_true == c) for c in classes])
        if np.sum(weights) == 0:
            return 0.0
        return float(np.average(f1s, weights=weights))
    return f1s


# ============================================================================
# Confusion Matrix
# ============================================================================

def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """Compute a confusion matrix from scratch.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    labels : list of int, optional
        Ordered list of class labels.  If ``None``, inferred from the data.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape ``(n_classes, n_classes)`` where entry
        ``[i, j]`` is the count of samples with true label ``i`` predicted
        as label ``j``.

    Examples
    --------
    >>> confusion_matrix(np.array([1, 1, 2, 2]), np.array([1, 2, 1, 2]))
    array([[1, 1],
           [1, 1]])
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    _check_shapes(y_true, y_pred)

    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())

    n = len(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        i = label_to_idx.get(true_label)
        j = label_to_idx.get(pred_label)
        if i is not None and j is not None:
            cm[i, j] += 1

    return cm


# ============================================================================
# Classification Report
# ============================================================================

def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """Generate a formatted classification report (similar to sklearn).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Predicted labels.
    target_names : list of str, optional
        Display names for each class.

    Returns
    -------
    str
        Formatted multi-line report string.

    Examples
    --------
    >>> print(classification_report(y_true, y_pred, target_names=["Walk", "Sit"]))
                  precision    recall  f1-score   support
    ...
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    _check_shapes(y_true, y_pred)
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())

    if target_names is None:
        target_names = [str(c) for c in classes]
    elif len(target_names) != len(classes):
        # Graceful fallback: if target_names count doesn't match the actual
        # number of classes in the data, use numeric labels instead of
        # silently truncating via zip.
        target_names = [str(c) for c in classes]

    # Header
    header = f"{'':>18s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}"
    lines = [header, "-" * len(header)]

    # Per-class rows
    for cls_label, name in zip(classes, target_names):
        p = precision(y_true, y_pred, cls_label)
        r = recall(y_true, y_pred, cls_label)
        f1 = f1_score(y_true, y_pred, cls_label)
        support = int(np.sum(y_true == cls_label))
        lines.append(f"{name:>18s} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {support:>10d}")

    # Summary rows
    lines.append("-" * len(header))

    macro_p = precision(y_true, y_pred, average="macro")
    macro_r = recall(y_true, y_pred, average="macro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    total_support = len(y_true)

    weighted_p = precision(y_true, y_pred, average="weighted")
    weighted_r = recall(y_true, y_pred, average="weighted")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    acc = accuracy(y_true, y_pred)

    lines.append(f"{'accuracy':>18s} {'':>10s} {'':>10s} {acc:>10.4f} {total_support:>10d}")
    lines.append(f"{'macro avg':>18s} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {total_support:>10d}")
    lines.append(f"{'weighted avg':>18s} {weighted_p:>10.4f} {weighted_r:>10.4f} {weighted_f1:>10.4f} {total_support:>10d}")

    return "\n".join(lines)
