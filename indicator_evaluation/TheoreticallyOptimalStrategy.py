from util import get_data, plot_data
import datetime as dt
import pandas as pd
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt


def testPolicy(symbol='AAPL', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    df = get_data([symbol], pd.date_range(sd, ed))
    price_df = df[[symbol]].ffill().bfill()
    df_trades = df[['SPY']]
    df_trades = df_trades.rename(
        columns={'SPY': symbol}).astype({symbol: 'int32'})
    df_trades[:] = 0
    dates = df_trades.index
    current_position = 0
    for i in range(len(dates) - 1):
        action = 1000 - current_position if (
            price_df.loc[dates[i+1]].loc[symbol] > price_df.loc[dates[i]].loc[symbol]) else -1000 - current_position
        df_trades.loc[dates[i]].loc[symbol] = action
        current_position += action
    return df_trades


def author():
    return 'dpanirwala3'
