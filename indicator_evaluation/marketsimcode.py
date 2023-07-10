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


def author():
    return 'dpanirwala3'


if __name__ == "__main__":
    pass
