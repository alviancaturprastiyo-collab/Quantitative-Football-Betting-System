import numpy as np
from qfbs.models.poisson import score_matrix, market_probabilities


def test_score_matrix_normalized():
    m = score_matrix(1.4, 1.1, 12)
    assert m.shape == (13, 13)
    assert np.isclose(m.sum(), 1.0)


def test_equal_lambdas_home_away_symmetry():
    m = score_matrix(1.2, 1.2, 12)
    p = market_probabilities(m)
    assert abs(p["home"] - p["away"]) < 1e-10


def test_market_probabilities_sum():
    p = market_probabilities(score_matrix(1.5, 0.9))
    assert np.isclose(p["home"] + p["draw"] + p["away"], 1.0)
    assert np.isclose(p["over_2_5"] + p["under_2_5"], 1.0)
