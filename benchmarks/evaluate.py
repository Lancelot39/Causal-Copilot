"""Evaluation metrics for causal discovery benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiscoveryMetrics:
    """Metrics comparing discovered adjacency to ground truth."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    shd: int  # Structural Hamming Distance


def evaluate_adjacency(predicted: np.ndarray, ground_truth: np.ndarray) -> DiscoveryMetrics:
    """Compare predicted adjacency matrix against ground truth.

    Both matrices use convention: mat[i,j]=1 means j→i.
    Predicted values >0 are treated as edges (handles 1/2/3 encoding).
    """
    pred_binary = (predicted > 0).astype(int)
    gt_binary = (ground_truth > 0).astype(int)

    tp = int(np.sum((pred_binary == 1) & (gt_binary == 1)))
    fp = int(np.sum((pred_binary == 1) & (gt_binary == 0)))
    fn = int(np.sum((pred_binary == 0) & (gt_binary == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # SHD = |edges in pred but not gt| + |edges in gt but not pred| + |reversed edges|
    shd = int(np.sum(pred_binary != gt_binary))

    return DiscoveryMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        shd=shd,
    )
