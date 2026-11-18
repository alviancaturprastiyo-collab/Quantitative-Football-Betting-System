"""Transparent decimal-odds value and Kelly calculations."""


def _validate(probability: float, odds: float) -> None:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if odds <= 1:
        raise ValueError("decimal odds must be greater than 1")


def implied_probability(odds: float) -> float:
    if odds <= 1:
        raise ValueError("decimal odds must be greater than 1")
    return 1.0 / odds


def expected_value(probability: float, odds: float) -> float:
    """Expected net profit per 1-unit stake: p*(odds-1) - (1-p)."""
    _validate(probability, odds)
    return probability * odds - 1.0


def kelly_fraction(probability: float, odds: float, fraction: float = 1.0) -> float:
    """Kelly stake fraction, floored at zero and optionally fractionally scaled."""
    _validate(probability, odds)
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    b = odds - 1.0
    full = (probability * odds - 1.0) / b
    return max(0.0, full) * fraction


def no_vig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Normalize two reciprocal prices to remove the overround proportionally."""
    if odds_a <= 1 or odds_b <= 1:
        raise ValueError("decimal odds must be greater than 1")
    a, b = 1 / odds_a, 1 / odds_b
    total = a + b
    return a / total, b / total
