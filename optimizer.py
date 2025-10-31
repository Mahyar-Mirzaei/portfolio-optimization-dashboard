import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List

def minimize_variance(cov_matrix: np.ndarray, bounds: List[Tuple[float, float]]=None, allow_short: bool=False):
    n = cov_matrix.shape[0]
    x0 = np.repeat(1/n, n)
    if bounds is None:
        if allow_short:
            bounds = [(-1.0, 1.0)] * n
        else:
            bounds = [(0.0, 1.0)] * n

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    def fun(w):
        return float(w.T @ cov_matrix @ w)

    res = minimize(fun, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return res

def max_sharpe(mean_returns: np.ndarray, cov_matrix: np.ndarray, risk_free: float = 0.0, bounds: List[Tuple[float, float]] = None, allow_short: bool=False):
    n = len(mean_returns)
    x0 = np.repeat(1/n, n)
    if bounds is None:
        if allow_short:
            bounds = [(-1.0, 1.0)] * n
        else:
            bounds = [(0.0, 1.0)] * n

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

    def neg_sharpe(w):
        ret = float(np.dot(w, mean_returns))
        vol = float(np.sqrt(w.T @ cov_matrix @ w))
        # if vol is zero, penalize
        if vol == 0: return 1e6
        return - (ret - risk_free) / vol

    res = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=cons)
    return res

def efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray, returns_grid: np.ndarray, bounds=None, allow_short=False):
    """
    For each target return in returns_grid, find the min variance portfolio meeting that return.
    Returns list of dict { 'weights', 'return', 'vol' }
    """
    n = len(mean_returns)
    x0 = np.repeat(1/n, n)
    if bounds is None:
        bounds = [(-1.0,1.0)]*n if allow_short else [(0.0,1.0)]*n

    base_cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    results = []
    for target in returns_grid:
        cons = base_cons + [{'type': 'eq', 'fun': lambda w, m=mean_returns, t=target: float(np.dot(w, m) - t)}]
        res = minimize(lambda w: float(w.T@cov_matrix@w), x0, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            w = res.x
            ret = float(np.dot(w, mean_returns))
            vol = float(np.sqrt(w.T @ cov_matrix @ w))
            results.append({'weights': w, 'return': ret, 'vol': vol})
    return results
