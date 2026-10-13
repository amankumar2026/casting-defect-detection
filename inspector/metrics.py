"""Scores for a defect detector, and the cost-based choice of threshold.

"Positive" always means defective:
    missed defect = a defective part predicted ok  (false negative)
    false reject  = a good part predicted defective (false positive)
"""

import math

import numpy as np

from inspector import config


def summarize(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    predicted = probs >= threshold
    defective = labels == 1
    caught = int((predicted & defective).sum())
    missed = int((~predicted & defective).sum())
    false_rejects = int((predicted & ~defective).sum())
    passed = int((~predicted & ~defective).sum())
    return {
        "threshold": round(float(threshold), 2),
        "n": len(labels),
        "accuracy": (caught + passed) / len(labels),
        "recall": caught / max(caught + missed, 1),            # share of defects caught
        "precision": caught / max(caught + false_rejects, 1),  # share of rejects that were defective
        "missed_defects": missed,
        "false_rejects": false_rejects,
        "cost": cost(missed, false_rejects),
    }


def cost(missed_defects: int, false_rejects: int) -> int:
    return (missed_defects * config.COST_MISSED_DEFECT
            + false_rejects * config.COST_FALSE_REJECT)


def pick_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    """The threshold with the lowest total cost.

    Thresholds 0.01 to 0.99 are tried. When several give the same lowest cost,
    the middle one is returned: it leaves the most room on both sides.
    """
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    costs = np.array([summarize(labels, probs, t)["cost"] for t in grid])
    best = grid[costs == costs.min()]
    return float(best[len(best) // 2])


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion (e.g. accuracy on n test photos)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    denominator = 1 + z * z / n
    return max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator)
