"""Independent Poisson score model and common market probabilities."""
from __future__ import annotations
import numpy as np
from scipy.stats import poisson


def score_matrix(home_lambda: float, away_lambda: float, max_goals: int = 10) -> np.ndarray:
    """Return P(home goals=i, away goals=j), truncated at max_goals.

    Lambdas must be non-negative. The returned matrix is normalized to account
    for the tiny probability mass above max_goals.
    """
    if home_lambda < 0 or away_lambda < 0:
        raise ValueError("Goal-rate lambdas must be non-negative")
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1")
    goals = np.arange(max_goals + 1)
    home = poisson.pmf(goals, home_lambda)
    away = poisson.pmf(goals, away_lambda)
    matrix = np.outer(home, away)
    total = matrix.sum()
    if total <= 0:
        raise ArithmeticError("Score probability matrix has zero mass")
    return matrix / total


def market_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """Derive 1X2, totals 2.5 and BTTS probabilities from a score matrix."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("Expected a square score matrix")
    if np.any(m < 0) or not np.isfinite(m).all() or m.sum() <= 0:
        raise ValueError("Matrix must contain finite non-negative probability mass")
    m = m / m.sum()
    home = np.tril(m, k=-1).sum()
    draw = np.trace(m)
    away = np.triu(m, k=1).sum()
    over25 = sum(m[i, j] for i in range(m.shape[0]) for j in range(m.shape[1]) if i + j >= 3)
    btts = m[1:, 1:].sum()
    return {
        "home": float(home), "draw": float(draw), "away": float(away),
        "over_2_5": float(over25), "under_2_5": float(1 - over25),
        "btts_yes": float(btts), "btts_no": float(1 - btts),
    }
