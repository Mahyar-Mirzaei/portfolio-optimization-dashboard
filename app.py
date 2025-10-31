import streamlit as st
import pandas as pd
import numpy as np
from src.data_fetch import read_prices_from_csv, fetch_prices_yfinance
from src.utils import compute_simple_returns, annualize_returns, annualize_cov, generate_random_portfolios, portfolio_performance
from src.optimizer import minimize_variance, max_sharpe, efficient_frontier
from src.backtest import rolling_rebalance_backtest
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimization Dashboard")

# --- Data input ---
st.sidebar.header("Data input")
use_upload = st.sidebar.checkbox("Upload prices CSV", value=True)
upload_file = None
prices = None

if use_upload:
    upload_file = st.sidebar.file_uploader("Upload CSV (Date column + ticker columns)", type=["csv"])
    if upload_file is not None:
        prices = read_prices_from_csv(upload_file)
else:
    # simple manual ticker fetch (optional)
    tickers_input = st.sidebar.text_input("Tickers (comma separated)", value="AAPL,MSFT,GOOG,AMZN")
    start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2018-01-01"))
    end_date = st.sidebar.date_input("End date", value=pd.to_datetime("2023-12-31"))
    if st.sidebar.button("Fetch with yfinance (optional)"):
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        try:
            prices = fetch_prices_yfinance(tickers, str(start_date), str(end_date))
            st.sidebar.write("Fetched prices for: " + ", ".join(prices.columns.tolist()))
        except Exception as e:
            st.sidebar.error(f"Failed to fetch: {e}")

if prices is None:
    st.info("Upload a CSV with a Date column and one column per asset (Adjusted Close). See README for format.")
    st.stop()

# --- Controls ---
st.sidebar.header("Settings")
start = st.sidebar.date_input("Plot start date", value=prices.index.min().date())
end = st.sidebar.date_input("Plot end date", value=prices.index.max().date())
rf = st.sidebar.number_input("Risk-free rate (annual)", value=0.02, step=0.001)
n_portfolios = st.sidebar.slider("Random portfolios", 100, 20000, 3000)
allow_short = st.sidebar.checkbox("Allow shorting (risky)", value=False)
rebalancing_days = st.sidebar.selectbox("Backtest rebalance period (days)", [30, 90, 180, 365], index=1)
lookback_days = st.sidebar.number_input("Lookback window (days) for backtest", value=252*3)
transaction_cost = st.sidebar.number_input("Transaction cost per turnover fraction", value=0.001)

# filter date
prices = prices.loc[pd.to_datetime(start):pd.to_datetime(end)]
st.subheader("Price chart")
fig = px.line(prices, labels={'value':'Price', 'index':'Date'}, title="Adjusted Close Prices")
st.plotly_chart(fig, use_container_width=True)

# compute returns and stats
daily_returns = compute_simple_returns(prices)
ann_returns = annualize_returns(daily_returns)
ann_cov = annualize_cov(daily_returns)

st.subheader("Summary statistics")
cols = st.columns(3)
cols[0].metric("Assets", len(prices.columns))
cols[1].metric("Start", str(prices.index.min().date()))
cols[2].metric("End", str(prices.index.max().date()))

st.write("Annualized returns (sample):")
st.dataframe(ann_returns.sort_values(ascending=False).to_frame("Annual Return"))

# random portfolios
st.subheader("Random portfolios & Efficient frontier")
results, weights_record = generate_random_portfolios(ann_returns.values, ann_cov.values, n_portfolios=n_portfolios, risk_free=rf)
df_rand = pd.DataFrame(results, columns=["Return", "Volatility", "Sharpe"])
df_rand["Sharpe"] = df_rand["Sharpe"]
fig2 = px.scatter(df_rand, x="Volatility", y="Return", color="Sharpe", title="Random Portfolios (Vol vs Return)", hover_data=[df_rand.index])
st.plotly_chart(fig2, use_container_width=True)

# optimization
st.subheader("Optimized portfolios")
n = len(ann_returns)
bounds = None
if allow_short:
    bounds = [(-1.0, 1.0)] * n
else:
    bounds = [(0.0, 1.0)] * n

# max sharpe
res_sharpe = max_sharpe(ann_returns.values, ann_cov.values, risk_free=rf, bounds=bounds, allow_short=allow_short)
w_sharpe = res_sharpe.x if res_sharpe.success else None
ret_sharpe, vol_sharpe, sharpe_sharpe = portfolio_performance(w_sharpe, ann_returns.values, ann_cov.values, rf) if w_sharpe is not None else (None, None, None)

# min var
res_minvar = minimize_variance(ann_cov.values, bounds=bounds, allow_short=allow_short)
w_minvar = res_minvar.x if res_minvar.success else None
ret_minvar, vol_minvar, sharpe_minvar = portfolio_performance(w_minvar, ann_returns.values, ann_cov.values, rf) if w_minvar is not None else (None, None, None)

st.write("Max Sharpe portfolio:")
if w_sharpe is not None:
    df_ws = pd.DataFrame({'Ticker': prices.columns, 'Weight': np.round(w_sharpe,4)})
    st.dataframe(df_ws.sort_values('Weight', ascending=False))
    st.write(f"Return: {ret_sharpe:.3f}, Vol: {vol_sharpe:.3f}, Sharpe: {sharpe_sharpe:.3f}")
else:
    st.write("Optimization failed.")

st.write("Min variance portfolio:")
if w_minvar is not None:
    df_wm = pd.DataFrame({'Ticker': prices.columns, 'Weight': np.round(w_minvar,4)})
    st.dataframe(df_wm.sort_values('Weight', ascending=False))
    st.write(f"Return: {ret_minvar:.3f}, Vol: {vol_minvar:.3f}, Sharpe: {sharpe_minvar:.3f}")
else:
    st.write("Optimization failed.")

# efficient frontier line
ret_min = df_rand["Return"].min()
ret_max = df_rand["Return"].max()
returns_grid = np.linspace(ret_min, ret_max, 50)
ef_results = efficient_frontier(ann_returns.values, ann_cov.values, returns_grid, bounds=bounds, allow_short=allow_short)

if ef_results:
    ef_df = pd.DataFrame(ef_results)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_rand['Volatility'], y=df_rand['Return'], mode='markers', marker=dict(color=df_rand['Sharpe'], showscale=True), name='Random portfolios'))
    fig3.add_trace(go.Scatter(x=ef_df['vol'], y=ef_df['return'], mode='lines', name='Efficient frontier', line=dict(color='black', width=2)))
    # add optimized points
    if w_sharpe is not None:
        fig3.add_trace(go.Scatter(x=[vol_sharpe], y=[ret_sharpe], mode='markers', name='Max Sharpe', marker=dict(size=12, color='red')))
    if w_minvar is not None:
        fig3.add_trace(go.Scatter(x=[vol_minvar], y=[ret_minvar], mode='markers', name='Min Var', marker=dict(size=12, color='green')))
    fig3.update_layout(title="Efficient Frontier")
    st.plotly_chart(fig3, use_container_width=True)

# backtest example
st.subheader("Backtest example (rolling rebalancing)")
def weight_fn_max_sharpe(history_prices):
    # compute returns & stats
    hist_rets = compute_simple_returns(history_prices)
    ann_ret = annualize_returns(hist_rets).values
    ann_cov_local = annualize_cov(hist_rets).values
    res = max_sharpe(ann_ret, ann_cov_local, risk_free=rf, bounds=bounds, allow_short=allow_short)
    return res.x if res.success else np.repeat(1.0/len(ann_ret), len(ann_ret))

if st.button("Run backtest (may take a while)"):
    with st.spinner("Running backtest..."):
        pv_df = rolling_rebalance_backtest(prices, lookback_days=lookback_days, rebalance_period_days=rebalancing_days, weight_fn=weight_fn_max_sharpe, transaction_cost=transaction_cost)
        fig_bt = px.line(pv_df, y='pv', title='Portfolio value over time (backtest)')
        st.plotly_chart(fig_bt, use_container_width=True)
        st.success("Backtest complete.")
