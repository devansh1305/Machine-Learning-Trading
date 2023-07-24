""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
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




import datetime as dt
import random
import pandas as pd
import numpy as np
import util as ut
import random
import BagLearner as bal
import RTLearner as rtl
from indicators import bollingerbands, PriceSMA, MACD
from datetime import datetime, timedelta
from util import get_data
class StrategyLearner(object):

    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		  		 			  		 			 	 	 		 		 	
        Constructor method  		  	   		  		 			  		 			 	 	 		 		 	
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = bal.BagLearner(learner=rtl.RTLearner, verbose=False, kwargs={
            "leaf_size": 5}, bags=40, boost=False)

    def author(self):
        return 'dpanirwala3'

    def calculate_yval(self, price):
        return (price.shift(-10) - price) / price

    def calculate_mean_std(self, ydata):
        # Calculate mean and StdDev of ydata
        mean = ydata.mean()
        std = ydata.std()
        return mean, std

    def calculate_buy_limit_sell_limit(self, mean, std):
        # Calculate buy_limit and sell_limit based on mean and StdDev
        buy_limit = mean + std * 0.5
        sell_limit = mean - std * 0.5
        return buy_limit, sell_limit

    def create_train_dataframe(self, price, sd, ed, symbol, lookback, impact):
        ydata = self.calculate_yval(price)
        train = pd.concat([
            price,
            bollingerbands(sd, ed, symbol, window_size=lookback),
            MACD(sd, ed, symbol),
            PriceSMA(sd, ed, symbol, window_size=lookback),
            ydata
        ], axis=1)
        train.columns = [symbol, 'bbp', 'MACD_Ratio', 'PriceOverSMA', 'Ydata']
        mean, std = self.calculate_mean_std(ydata)
        buy_limit, sell_limit = self.calculate_buy_limit_sell_limit(mean, std)
        train['Y'] = np.select([
            (train['Ydata'] - impact > buy_limit),
            (train['Ydata'] + impact < sell_limit)
        ], [1, -1], default=0)
        train.dropna(axis=0, how='any', inplace=True)
        return train.iloc[:, 0:-2].values, train.iloc[:, -1].values

    def add_evidence(
        self,
        symbol="IBM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 1, 1),
        sv=10000,
    ):
        lookback = 20
        new_sd = sd - dt.timedelta(days=lookback+15)
        df = get_data([symbol], pd.date_range(new_sd, ed))
        price = df[symbol]
        price = price.loc[sd:ed]

        # print price

        train = pd.concat(
            [price,
             bollingerbands(sd, ed, symbol, window_size=lookback),
             MACD(sd, ed, symbol),
             PriceSMA(sd, ed, symbol, window_size=lookback),
             self.calculate_yval(price)], axis=1)
        trainX, trainY = self.create_train_dataframe(
            price, sd, ed, symbol, lookback, self.impact)

        self.learner.add_evidence(trainX, trainY)

    def testPolicy(self, symbol="IBM",
                   sd=dt.datetime(2009, 1, 1),
                   ed=dt.datetime(2010, 1, 1),
                   sv=10000):

        lookback = 20
        sym = [symbol]
        new_sd = sd - timedelta(days=lookback+15)
        df = get_data([symbol], pd.date_range(new_sd, ed))
        price = df[symbol]
        price = price.loc[sd:ed]
        testX = pd.concat([price,
                           bollingerbands(
                               sd, ed, symbol, window_size=lookback),
                           MACD(sd, ed, symbol),
                           PriceSMA(sd, ed, symbol, window_size=lookback)], axis=1)
        testX.columns = [symbol, 'bbp', 'MACD_Ratio', 'PriceOverSMA']

        testX = testX.values

        queryY = self.learner.query(testX)
        df_dates = df[sym].loc[sd:ed]
        trades = df_dates.copy()
        trades.ix[:] = 0.0
        cashBalance = 0
        trade_actions = {
            1.0: {0: (1000, 1000), 1000: (0, 0), -1000: (2000, 2000)},
            -1.0: {0: (-1000, -1000), -1000: (0, 0), 1000: (-2000, -2000)},
        }
        for i in range(df_dates.shape[0] - 1):
            action = queryY[i]

            if action in trade_actions:
                if cashBalance in trade_actions[action]:
                    trade_value, balance_update = trade_actions[action][cashBalance]
                    trades.ix[i, symbol] = trade_value
                    cashBalance += balance_update

        return trades


def author():
    return 'dpanirwala3'


if __name__ == "__main__":
    print("One does not simply think up a strategy")
