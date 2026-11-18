"""Proper scoring rules for probability forecasts."""
import numpy as np


def brier_score(probabilities, outcomes) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probabilities and outcomes must have equal non-empty shapes")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    return float(np.mean((p - y) ** 2))


def log_loss(probabilities, outcomes, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1 - eps)
    y = np.asarray(outcomes, dtype=float)
    if p.shape != y.shape or p.size == 0:
        raise ValueError("probabilities and outcomes must have equal non-empty shapes")
    if np.any((y < 0) | (y > 1)):
        raise ValueError("outcomes must be in [0, 1]")
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
