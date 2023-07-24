import matplotlib.pyplot as plt
import pandas as pd
import warnings
import datetime as dt
import numpy as np
import marketsimcode as msc
import ManualStrategy as ms
import StrategyLearner as sl
import random
from datetime import datetime, timedelta
from util import get_data, plot_data
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def author():
    return 'dpanirwala3'


def run_manual_strategy(sym, in_sample_start_dt, in_sample_end_dt, startval):
    mss = ms.ManualStrategy()
    long, short, df_trades = mss.testPolicy(
        sym, in_sample_start_dt, in_sample_end_dt, startval)

    portvals = msc.compute_portvals(df_trades, startval, 0.0, 0.0)
    returns_cumu = (portvals[-1] / portvals[0]) - 1
    daily_returns = portvals / portvals.shift(1) - 1
    daily_returns = daily_returns[1:]
    ADR = daily_returns.mean()
    stddev_returns = daily_returns.std()
    K = np.sqrt(252.0)
    sharpe_ratio = K * (ADR / stddev_returns)

    return portvals, returns_cumu, stddev_returns, ADR, sharpe_ratio


def run_strategy_learner(sym, in_sample_start_dt, in_sample_end_dt, startval):
    np.random.seed(13)
    random.seed(13)
    stl = sl.StrategyLearner(verbose=False, impact=0.0)
    stl.add_evidence(sym, dt.datetime(2008, 1, 1),
                     dt.datetime(2009, 12, 31), startval)
    trades = stl.testPolicy(sym, in_sample_start_dt,
                            in_sample_end_dt, startval)
    st_portvals = msc.compute_portvals(trades, startval, 0.0, 0.0)
    st_cum_ret = (st_portvals[-1] / st_portvals[0]) - 1
    st_daily_returns = st_portvals / st_portvals.shift(1) - 1
    st_daily_returns = st_daily_returns[1:]
    st_avg_daily_ret = st_daily_returns.mean()
    st_std_daily_ret = st_daily_returns.std()
    K = np.sqrt(252.0)
    st_sharpe_ratio = K * (st_avg_daily_ret / st_std_daily_ret)

    return st_portvals, st_cum_ret, st_std_daily_ret, st_avg_daily_ret, st_sharpe_ratio


def run_benchmark(sym, in_sample_start_dt, in_sample_end_dt, startval):
    df = get_data([sym], pd.date_range(
        in_sample_start_dt - timedelta(days=35), in_sample_end_dt))
    bench_trades = df[[sym]].loc[in_sample_start_dt:in_sample_end_dt]
    bench_trades.ix[:] = 0.0
    bench_trades.ix[0, sym] = 1000
    bench_trades.ix[-1, sym] = -1000

    bench_port = msc.compute_portvals(bench_trades, startval, 0.0, 0.0)
    bench_cumu = (bench_port[-1] / bench_port[0]) - 1
    bench_DR = bench_port / bench_port.shift(1) - 1
    bench_DR = bench_DR[1:]
    bench_ADR = bench_DR.mean()
    bench_stddev = bench_DR.std()
    K = np.sqrt(252.0)
    sharpe_ratio_bnch = K * (bench_ADR / bench_stddev)

    return bench_port, bench_cumu, bench_stddev, bench_ADR, sharpe_ratio_bnch


def plot_results_in_sample(portval_norm_fund, bnch_norm_fund, strategy_norm_fund, in_sample_start_dt, in_sample_end_dt):
    plt.plot(portval_norm_fund, color='Blue', label='Manual Strategy')
    plt.plot(bnch_norm_fund, color='Red', label='Benchmark')
    plt.plot(strategy_norm_fund, color='Green', label='ML Strategy')

    plt.legend(fontsize=16)
    plt.title('Manual Strategy Vs ML Strategy Vs Benchmark', fontsize=16)
    plt.xlim(in_sample_start_dt, in_sample_end_dt)
    plt.xticks(rotation=25)
    plt.ylabel('Value', fontsize=16)
    plt.xlabel('Dates', fontsize=16)

    plt.savefig('images/Exp1_in_sample.png')


def plot_results(title, portval_norm_fund, bnch_norm_fund, strategy_norm_fund, in_sample_start_dt, in_sample_end_dt):
    plt.clf()
    plt.plot(portval_norm_fund, color='Blue', label='Manual Strategy')
    plt.plot(bnch_norm_fund, color='Red', label='Benchmark')
    plt.plot(strategy_norm_fund, color='Green', label='ML Strategy')

    plt.legend(fontsize=16)
    plt.title('Strategy Comparison for '+title, fontsize=10)
    plt.xlim(in_sample_start_dt, in_sample_end_dt)
    plt.xticks(rotation=25)
    plt.ylabel('Value', fontsize=16)
    plt.xlabel('Dates', fontsize=16)

    plt.savefig("images/"+title+".png")


def testStrategy():
    sym = 'JPM'
    in_sample_start_dt = dt.datetime(2008, 1, 1)
    in_sample_end_dt = dt.datetime(2009, 12, 31)
    startval = 100000

    portvals, returns_cumu, stddev_returns, ADR, sharpe_ratio = run_manual_strategy(
        sym, in_sample_start_dt, in_sample_end_dt, startval)
    st_portvals, st_cum_ret, st_std_daily_ret, st_avg_daily_ret, st_sharpe_ratio = run_strategy_learner(
        sym, in_sample_start_dt, in_sample_end_dt, startval)
    bench_port, bench_cumu, bench_stddev, bench_ADR, sharpe_ratio_bnch = run_benchmark(
        sym, in_sample_start_dt, in_sample_end_dt, startval)

    print()
    print("In sample Date Range: {} to {}".format(
        in_sample_start_dt, in_sample_end_dt))
    print()
    print("In-Sample Cumulative Return in Manual Strategy: {}".format(returns_cumu))
    print("In-sample cumulative return in Benchmark : {}".format(bench_cumu))
    print("In-sample cumulative return in Strategy Learner : {}".format(st_cum_ret))
    print()
    print("In-sample StdDev in Manual Strategy: {}".format(stddev_returns))
    print("In-Sample StdDev in Benchmark : {}".format(bench_stddev))
    print("In-Sample StdDev in Strategy Learner : {}".format(st_std_daily_ret))
    print()
    print("In-Sample ADR in Manual Strategy: {}".format(ADR))
    print("In-Sample ADR in Benchmark : {}".format(bench_ADR))
    print("In-Sample ADR in Strategy Learner: {}".format(st_avg_daily_ret))
    print()
    print("In-Sample Sharpe Ratio in Manual Strategy: {}".format(sharpe_ratio))
    print("In-Sample Sharpe Ratio in Strategy Learner: {}".format(st_sharpe_ratio))
    print("In-Sample Sharpe Ratio in benchmark : {}".format(sharpe_ratio_bnch))
    print()
    print(
        "In-Sample Ending Portfolio Value in Manual Strategy: {}".format(portvals[-1]))
    print(
        "In-Sample Ending Portfolio value in benchmark :{}".format(bench_port[-1]))
    print(
        "In-Sample Ending Portfolio value in Strategy Learner :{}".format(st_portvals[-1]))
    print()
    print('**************************************************************************')
    print()

    portval_norm_fund = portvals / portvals.ix[0,]
    bnch_norm_fund = bench_port / bench_port.ix[0,]
    strategy_norm_fund = st_portvals / st_portvals.ix[0,]
    plot_results("Exp1_in_sample", portval_norm_fund, bnch_norm_fund,
                 strategy_norm_fund, in_sample_start_dt, in_sample_end_dt)

    sym = 'JPM'
    out_sample_start_dt = dt.datetime(2010, 1, 1)
    out_sample_end_dt = dt.datetime(2011, 12, 31)
    startval = 100000

    portvals, returns_cumu, stddev_returns, ADR, sharpe_ratio = run_manual_strategy(
        sym, out_sample_start_dt, out_sample_end_dt, startval)
    st_portvals, st_cum_ret, st_std_daily_ret, st_avg_daily_ret, st_sharpe_ratio = run_strategy_learner(
        sym, out_sample_start_dt, out_sample_end_dt, startval)
    bench_port, bench_cumu, bench_stddev, bench_ADR, sharpe_ratio_bnch = run_benchmark(
        sym, out_sample_start_dt, out_sample_end_dt, startval)

    print()
    print("Out sample Date Range: {} to {}".format(
        out_sample_start_dt, out_sample_end_dt))
    print()
    print("Out-Sample Cumulative Return in Manual Strategy: {}".format(returns_cumu))
    print("Out-sample cumulative return in Benchmark : {}".format(bench_cumu))
    print("Out-sample cumulative return in Strategy Learner : {}".format(st_cum_ret))
    print()
    print("Out-sample StdDev in Manual Strategy: {}".format(stddev_returns))
    print("Out-Sample StdDev in Benchmark : {}".format(bench_stddev))
    print("Out-Sample StdDev in Strategy Learner : {}".format(st_std_daily_ret))
    print()
    print("Out-Sample ADR in Manual Strategy: {}".format(ADR))
    print("Out-Sample ADR in Benchmark : {}".format(bench_ADR))
    print("Out-Sample ADR in Strategy Learner: {}".format(st_avg_daily_ret))
    print()
    print("Out-Sample Sharpe Ratio in Manual Strategy: {}".format(sharpe_ratio))
    print("Out-Sample Sharpe Ratio in Strategy Learner: {}".format(st_sharpe_ratio))
    print("Out-Sample Sharpe Ratio in benchmark : {}".format(sharpe_ratio_bnch))
    print()
    print(
        "Out-Sample Ending Portfolio Value in Manual Strategy: {}".format(portvals[-1]))
    print(
        "Out-Sample Ending Portfolio value in benchmark :{}".format(bench_port[-1]))
    print(
        "Out-Sample Ending Portfolio value in Strategy Learner :{}".format(st_portvals[-1]))
    print()
    print('**************************************************************************')
    print()

    portval_norm_fund = portvals / portvals.ix[0,]
    bnch_norm_fund = bench_port / bench_port.ix[0,]
    strategy_norm_fund = st_portvals / st_portvals.ix[0,]
    plot_results("Exp1_out_sample", portval_norm_fund, bnch_norm_fund,
                 strategy_norm_fund, out_sample_start_dt, out_sample_end_dt)


if __name__ == "__main__":
    testStrategy()
