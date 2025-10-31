import numpy as np
import pandas as pd
from typing import Tuple, Callable

def rolling_rebalance_backtest(prices: pd.DataFrame,
                               lookback_days: int,
                               rebalance_period_days: int,
                               weight_fn: Callable[[pd.DataFrame], np.ndarray],
                               transaction_cost: float = 0.0):
    """
    prices: DataFrame of prices (index Datetime)
    lookback_days: window for estimating mean/cov
    rebalance_period_days: how often to rebalance
    weight_fn: function that receives a price DataFrame of lookback window and returns weights (numpy array)
    transaction_cost: fraction cost per turnover (e.g., 0.001 = 0.1%)
    Returns: DataFrame of portfolio value over time
    """
    prices = prices.sort_index()
    returns = prices.pct_change().dropna()
    dates = returns.index
    # initial portfolio value
    pv = 1.0
    pv_history = []
    current_weights = None
    last_rebalance = dates[0]

    # iterate daily and rebalance when needed
    for i, date in enumerate(dates):
        # check if we should rebalance (first day and every rebalance_period_days)
        if (date - last_rebalance).days >= rebalance_period_days or current_weights is None:
            # define lookback window
            start_look = date - pd.Timedelta(days=lookback_days)
            lookback_prices = prices[(prices.index >= start_look) & (prices.index < date)]
            if len(lookback_prices) < 2:
                # can't compute, skip
                pass
            else:
                new_weights = weight_fn(lookback_prices)
                # apply transaction cost proportional to turnover
                if current_weights is not None:
                    turnover = np.sum(np.abs(new_weights - current_weights))
                    pv *= (1 - transaction_cost * turnover)
                current_weights = new_weights
                last_rebalance = date

        # apply daily return to portfolio value using current weights
        if current_weights is None:
            pv_history.append({'date': date, 'pv': pv})
            continue
        # today's returns vector
        r = returns.loc[date].values
        portfolio_ret = float(np.dot(current_weights, r))
        pv = pv * (1 + portfolio_ret)
        pv_history.append({'date': date, 'pv': pv})

    df = pd.DataFrame(pv_history).set_index('date')
    return df
