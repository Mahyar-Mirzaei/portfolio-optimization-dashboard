import numpy as np
import pandas as pd

def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple returns from adjusted close price DataFrame.
    Input: DataFrame indexed by Date with columns as tickers/prices.
    """
    return prices.pct_change().dropna()

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

def annualize_returns(daily_returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    mean_daily = daily_returns.mean()
    return mean_daily * periods_per_year

def annualize_cov(daily_returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    return daily_returns.cov() * periods_per_year

def portfolio_performance(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free: float = 0.0):
    """
    Returns (annual_return, annual_volatility, sharpe)
    mean_returns: array of annualized expected returns (same ordering as weights)
    cov_matrix: annualized covariance matrix
    """
    ret = float(np.dot(weights, mean_returns))
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))
    sharpe = (ret - risk_free) / vol if vol != 0 else 0.0
    return ret, vol, sharpe

def generate_random_portfolios(mean_returns: np.ndarray, cov_matrix: np.ndarray, n_portfolios: int = 5000, risk_free: float = 0.0, seed: int = 42):
    """
    Generates random long-only portfolios (weights summing to 1).
    Returns: results array shape (n_portfolios, 3) for [return, vol, sharpe] and list of weight arrays.
    """
    np.random.seed(seed)
    n = len(mean_returns)
    results = np.zeros((n_portfolios, 3))
    weights_record = []
    for i in range(n_portfolios):
        w = np.random.random(n)
        w /= np.sum(w)
        ret, vol, sharpe = portfolio_performance(w, mean_returns, cov_matrix, risk_free)
        results[i, :] = [ret, vol, sharpe]
        weights_record.append(w)
    return results, weights_record
