import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def compute_portvals(orders_df, start_val=1000000, commission=9.95, impact=0.005):

    orders_dates = orders_df.index
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]

    portfolioValues = get_data(['SPY'], pd.date_range(
        start_date, end_date), addSPY=True, colname='Adj Close')
    portfolioValues = portfolioValues.rename(columns={'SPY': 'value'})
    dates = portfolioValues.index

    symbol = orders_df.columns[0]

    def updatePortfolio(symbol, order, shares, balance, shares_owned, symbol_table, currentdate, end_date, commission, impact):
        if symbol not in symbol_table:
            symbol_table[symbol] = get_data([symbol], pd.date_range(
                currentdate, end_date), addSPY=True, colname='Adj Close').ffill().bfill()
        impactMovement = impact if (order == 'BUY') else -impact
        cashMovement = symbol_table[symbol].loc[currentdate].loc[symbol] * \
            (1 + impactMovement) * shares
        cashMovement = -cashMovement if (order == 'BUY') else cashMovement
        shareMovement = shares if (order == 'BUY') else -shares
        shares_owned[symbol] = shares_owned.get(symbol, 0) + shareMovement
        balance += cashMovement - commission
        return balance, shares_owned, symbol_table

    balance = start_val
    shares_owned = {}
    symbol_table = {}

    for date in dates:

        trade = orders_df.loc[date].loc[symbol]

        if trade != 0:
            order = 'SELL' if (trade < 0) else 'BUY'
            shares = abs(trade) if (trade < 0) else trade
            balance, shares_owned, symbol_table = \
                updatePortfolio(symbol, order, shares, balance, shares_owned,
                                symbol_table, date, end_date, commission, impact)

        shares_worth = 0
        for symbol in shares_owned:
            shares_worth += symbol_table[symbol].loc[date].loc[symbol] * \
                shares_owned[symbol]
        portfolioValues.loc[date].loc['value'] = balance + shares_worth

    return portfolioValues


def compute_portvals(df_trades, start_val=100000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    df_trades = df_trades.sort_index()

    start_dt = df_trades.index[0]
    end_dt = df_trades.index[-1]

    symbols = list(df_trades)

    dates = pd.date_range(start_dt, end_dt)

    df3 = get_data(symbols, dates)

    valid_days = df_trades.index.isin(df3.index.values)
    df_trades = df_trades[valid_days]

    df_trades['Cash'] = 0.0
    df3['Cash'] = 1.0

    df3.fillna(method='ffill', inplace=True)
    df3.fillna(method='bfill', inplace=True)

    sym = symbols[0]
    for index, row in df_trades.iterrows():
        # print row
        # prit df_trades.loc[index, sym]
        if (df_trades.loc[index, sym] > 0):
            df_trades.loc[index, "Cash"] = df_trades.loc[index, "Cash"] - df3.loc[index,
                                                                                  sym] * row[sym] - commission - abs(df3.loc[index, sym] * row[sym]) * impact
        elif (df_trades.loc[index, sym] < 0):
            df_trades.loc[index, "Cash"] = df_trades.loc[index, "Cash"] - df3.loc[index,
                                                                                  sym] * row[sym] - commission - abs(df3.loc[index, sym] * row[sym]) * impact

    df_holdings = df_trades.copy()
    df_holdings.ix[0, 'Cash'] = start_val + df_trades['Cash'][0]
    n = df_holdings.shape[0]
    for i in range(1, n):
        df_holdings.ix[i, :] += df_holdings.ix[i - 1, :]

    df_values = df_holdings.copy()
    df_values = df3 * df_holdings
    port_val = df_values.sum(axis=1)
    return port_val


def author():
    return 'dpanirwala3'


if __name__ == "__main__":
    pass
