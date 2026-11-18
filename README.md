# Quantitative Football Betting System (QFBS)

A clean, testable research codebase for football score modelling, market probabilities,
expected value, bankroll sizing and historical evaluation. **This is a research toolkit,
not a betting guarantee.** No model output guarantees an edge or profit.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## First calculation

```python
from qfbs.models.poisson import score_matrix, market_probabilities
from qfbs.betting.value import expected_value, kelly_fraction

matrix = score_matrix(home_lambda=1.55, away_lambda=1.10, max_goals=10)
markets = market_probabilities(matrix)
print(markets)  # home/draw/away, over 2.5, BTTS

p = markets["home"]
odds = 2.20
print("EV per unit:", expected_value(p, odds))
print("Kelly fraction:", kelly_fraction(p, odds))
```

## Architecture

- `qfbs/data`: CSV ingestion and schema validation
- `qfbs/models`: score models and parameter estimation
- `qfbs/probability`: market probability transformations
- `qfbs/betting`: odds, EV and conservative Kelly sizing
- `qfbs/backtesting`: chronological evaluation utilities
- `qfbs/live`: reserved for state-conditioned live models
- `notebooks`: Google Colab entry point
- `tests`: deterministic unit tests

## Data

Keep licensed/raw datasets in `data/raw/` (ignored by git). The initial CSV loader accepts
Football-Data-style columns but does not redistribute any third-party data. Confirm each
source's licence and terms before collecting or publishing data.

## Modelling notes

- Independent Poisson is the baseline, not a claim that goals are independent in reality.
- Dixon–Coles correction, Elo, xG and live state models are planned extensions.
- Evaluate with chronological train/test splits; never randomly shuffle match history.
- Report log loss, Brier score, calibration and closing-line value alongside ROI.
- Include bookmaker margin and realistic odds availability in backtests.
- Kelly is highly sensitive to probability error. Use fractional Kelly and strict exposure caps.

## GitHub Actions

The CI workflow installs the package and runs `pytest` on pushes and pull requests.
