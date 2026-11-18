import pytest
from qfbs.betting.value import expected_value, kelly_fraction, no_vig_two_way


def test_ev_and_kelly_positive_edge():
    assert expected_value(0.5, 2.2) == pytest.approx(0.1)
    assert kelly_fraction(0.5, 2.2) > 0


def test_no_edge_means_no_bet():
    assert kelly_fraction(0.4, 2.0) == 0


def test_no_vig_sums_to_one():
    a, b = no_vig_two_way(1.9, 1.9)
    assert a + b == pytest.approx(1.0)
    assert a == pytest.approx(0.5)
