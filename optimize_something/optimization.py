""""""
"""MC1-P2: Optimize a portfolio.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
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
  		  	   		  		 			  		 			 	 	 		 		 	
Student Name: Devansh Panirwala (replace with your name)  		  	   		  		 			  		 			 	 	 		 		 	
GT User ID: dpanirwala3 (replace with your User ID)  		  	   		  		 			  		 			 	 	 		 		 	
GT ID: 903262441 (replace with your GT ID)  		  	   		  		 			  		 			 	 	 		 		 	
"""


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality




import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from util import get_data, plot_data
def optimize_portfolio(
    sd=dt.datetime(2008, 1, 1),
    ed=dt.datetime(2009, 1, 1),
    syms=["GOOG", "AAPL", "GLD", "XOM"],
    gen_plot=False,
):
    """  		  	   		  		 			  		 			 	 	 		 		 	
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		  		 			  		 			 	 	 		 		 	
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  		 			  		 			 	 	 		 		 	
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		  		 			  		 			 	 	 		 		 	
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		  		 			  		 			 	 	 		 		 	
    statistics.  		  	   		  		 			  		 			 	 	 		 		 	

    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 			  		 			 	 	 		 		 	
    :type sd: datetime  		  	   		  		 			  		 			 	 	 		 		 	
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 			  		 			 	 	 		 		 	
    :type ed: datetime  		  	   		  		 			  		 			 	 	 		 		 	
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		  		 			  		 			 	 	 		 		 	
        symbol in the data directory)  		  	   		  		 			  		 			 	 	 		 		 	
    :type syms: list  		  	   		  		 			  		 			 	 	 		 		 	
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  		 			  		 			 	 	 		 		 	
        code with gen_plot = False.  		  	   		  		 			  		 			 	 	 		 		 	
    :type gen_plot: bool  		  	   		  		 			  		 			 	 	 		 		 	
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  		 			  		 			 	 	 		 		 	
        standard deviation of daily returns, and Sharpe ratio  		  	   		  		 			  		 			 	 	 		 		 	
    :rtype: tuple  		  	   		  		 			  		 			 	 	 		 		 	
    """

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    # print(prices_all)
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later
    log_returns = np.log(prices / prices.shift(1))
    weights = allocate_portfolio(log_returns)
    sddr, cr, adr, sr = get_portfolio_stats(log_returns, weights)
    if gen_plot:
        # add code to plot here
        normed = prices.div(prices.iloc[0])
        pos_val = normed * weights
        port_val = pos_val.sum(axis=1)
        df_temp = pd.concat(
            [port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1
        )
        df_temp = df_temp/df_temp.ix[0, :]
        df_temp.plot(title='Daily Portfolio Value and SPY')
        plt.legend(loc='lower left')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.grid(linewidth=1)
        plt.savefig('images/Figure1.png')
        pass

    return weights, cr, adr, sddr, sr


def sharpe_ratio(weights, log_returns):
    returns = np.sum(log_returns.mean() * weights)
    std_dev = np.sqrt(
        np.dot(weights.T, np.dot(log_returns.cov(), weights)))
    sharpe_ratio = returns / std_dev
    return sharpe_ratio


def allocate_portfolio(log_returns):
    num_assets = len(log_returns.columns)
    weights = np.ones(num_assets) / num_assets
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    result = minimize(lambda x: -sharpe_ratio(x, log_returns), weights,
                      method='SLSQP', bounds=tuple([(0, 1)]*num_assets), constraints=constraints)
    return result.x


def get_portfolio_stats(log_returns, weights):
    log_returns = np.sum(log_returns * weights)
    sddr = np.std(log_returns)
    cr = np.exp(np.cumsum(log_returns)) - 1
    adr = np.mean(log_returns)
    sharpe_ratio = np.sqrt(252) * adr / sddr
    return sddr, cr, adr, sharpe_ratio


def test_code():
    """  		  	   		  		 			  		 			 	 	 		 		 	
    This function WILL NOT be called by the auto grader.  		  	   		  		 			  		 			 	 	 		 		 	
    """

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )

    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
