import pandas as pd
from util import get_data, plot_data
import datetime as dt
import numpy as np
import marketsimcode as msc
import ManualStrategy as ms
import StrategyLearner as sl
import random
import matplotlib.pyplot as plt


def author():
    return 'dpanirwala3'


def run_strategy_learner_with_impact(sym, start_dt, end_dt, startval, impact):
    np.random.seed(13)
    random.seed(13)
    stl = sl.StrategyLearner(verbose=False, impact=impact)
    stl.add_evidence(sym, start_dt, end_dt, startval)
    trades = stl.testPolicy(sym, start_dt, end_dt, startval)

    seqTrade = trades[sym].astype(bool).sum(axis=0)
    st_portvals = msc.compute_portvals(trades, startval, 0, 0)
    st_daily_returns = (st_portvals / st_portvals.shift(1) - 1)[1:]
    st_sharpe_ratio = np.sqrt(
        252.0) * (st_daily_returns.mean() / st_daily_returns.std())

    return seqTrade, (st_portvals[-1] / st_portvals[0]) - 1, st_sharpe_ratio


def testcode():
    sym = 'JPM'
    start_dt = dt.datetime(2008, 1, 1)
    end_dt = dt.datetime(2009, 12, 31)
    startval = 100000

    impactlst = [0, 0.0005, 0.0015, 0.005, 0.01, 0.025, 0.05]
    yVal = []
    y_sharpe = []
    y_seqTrade = []
    portval = []

    for i in impactlst:
        seqTrade, st_cum_ret, st_sharpe_ratio = run_strategy_learner_with_impact(
            sym, start_dt, end_dt, startval, i)

        y_seqTrade.append(seqTrade)
        yVal.append(st_cum_ret)
        y_sharpe.append(st_sharpe_ratio)

    plt.figure(figsize=(15, 8))

    plt.plot(impactlst, y_seqTrade, label='Trade Quantity', color='Red')

    plt.legend(fontsize=16)
    plt.grid()

    plt.title('Trade Quantity VS impact', fontsize=16)
    plt.xticks(fontsize=16, rotation=25)
    plt.yticks(fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.xlabel('Impact', fontsize=16)

    plt.savefig('images/Exp2_b.png')

    plt.figure(figsize=(15, 8))

    plt.plot(impactlst, yVal, label='Cumulative Return', color='Red')

    plt.legend(fontsize=16)
    plt.grid()

    plt.title('Cumulative Return VS impact', fontsize=16)
    plt.xticks(fontsize=16, rotation=25)
    plt.yticks(fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.xlabel('Impact', fontsize=16)

    plt.savefig('images/Exp2_a.png')


if __name__ == "__main__":
    testcode()
