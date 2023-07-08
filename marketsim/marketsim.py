""""""
"""MC2-P1: Market simulator.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		  		 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			 	 	 		 		 	
or edited.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Student Name: Tucker Balch (replace with your name)  		  	   		  		 			  		 			 	 	 		 		 	
GT User ID: tb34 (replace with your User ID)  		  	   		  		 			  		 			 	 	 		 		 	
GT ID: 900897987 (replace with your GT ID)  		  	   		  		 			  		 			 	 	 		 		 	
"""




import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
def author():
    return "dpanirwala3"


def compute_portvals(
    orders_file="./orders/orders-01.csv",
    start_val=1000000,
    commission=9.95,
    impact=0.005,
):
    orders = pd.read_csv(orders_file, index_col='Date',
                         parse_dates=True, na_values=['nan'])
    orders_dates, start_date, end_date = orders.index, orders.index[
        0], orders.index[-1]

    portfolioValues = get_data(['SPY'], pd.date_range(
        start_date, end_date), addSPY=True, colname='Adj Close')
    portfolioValues = portfolioValues.rename(columns={'SPY': 'value'})
    dates = portfolioValues.index

    balance = start_val
    shares_owned = {}
    symbol_table = {}

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

    for date in dates:
        if date in orders_dates:
            data = orders.loc[date]

            if isinstance(data, pd.DataFrame):
                for _, each in data.iterrows():
                    balance, shares_owned, symbol_table = \
                        updatePortfolio(each.loc['Symbol'], each.loc['Order'], each.loc['Shares'], balance, shares_owned,
                                        symbol_table, date, end_date, commission, impact)
            else:
                balance, shares_owned, symbol_table = \
                    updatePortfolio(data.loc['Symbol'], data.loc['Order'], data.loc['Shares'], balance, shares_owned,
                                    symbol_table, date, end_date, commission, impact)

        shares_worth = 0
        for symbol in shares_owned:
            shares_worth += symbol_table[symbol].loc[date].loc[symbol] * \
                shares_owned[symbol]
        portfolioValues.loc[date].loc['value'] = balance + shares_worth

    return portfolioValues


def test_code():
    """  		  	   		  		 			  		 			 	 	 		 		 	
    Helper function to test code  		  	   		  		 			  		 			 	 	 		 		 	
    """
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./testcases_mc2p1/orders-04.csv"
    portfolioValues = compute_portvals(orders_file=of)
    if isinstance(portfolioValues, pd.DataFrame):
        portfolioValues = portfolioValues[portfolioValues.columns[0]]
    else:
        print("warning, no DataFrame")

    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    data = [
        0.2,
        0.01,
        0.02,
        1.5,
    ]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = data
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = data

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portfolioValues[-1]}")


if __name__ == "__main__":
    test_code()
