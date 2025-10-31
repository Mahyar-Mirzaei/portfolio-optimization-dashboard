import pandas as pd
import numpy as np
from src.utils import compute_simple_returns, annualize_returns, portfolio_performance

def test_returns_and_annualize():
    dates = pd.date_range("2020-01-01", periods=3, freq='D')
    df = pd.DataFrame({"A":[100, 110, 121], "B":[200, 210, 231]}, index=dates)
    rets = compute_simple_returns(df)
    assert rets.shape[0] == 2
    ann = annualize_returns(rets, periods_per_year=252)
    assert ann.shape[0] == 2

def test_portfolio_perf():
    mean = np.array([0.1, 0.2])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])
    w = np.array([0.5, 0.5])
    r, v, s = portfolio_performance(w, mean, cov, risk_free=0.01)
    assert r > 0
    assert v > 0
