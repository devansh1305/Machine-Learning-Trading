import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt
import datetime as dt
import marketsimcode as ms
import numpy as np
from indicators import bollingerbands, PriceSMA, MACD
from datetime import datetime, timedelta
import marketsimcode as msc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class ManualStrategy(object):
    def author(self):
        return 'dpanirwala3'

    def testPolicy(self, symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        lookback = 20

        # Calculate the required indicators
        bbp = bollingerbands(sd, ed, symbol, window_size=lookback)
        priceOverSMA = PriceSMA(sd, ed, symbol, window_size=lookback)
        macd = MACD(sd, ed, symbol).loc[sd:ed]

        # Get the price data for the given date range
        new_sd = sd - timedelta(days=lookback + 15)
        df = get_data([symbol], pd.date_range(new_sd, ed))
        df_dates = df[[symbol]].loc[sd:ed]
        d = df_dates.shape[0] - 1

        df_trades = df_dates.copy()
        df_trades.ix[:] = 0.0
        cashBalance = 0
        long_trades = []
        short_trades = []

        for i in range(d):
            if priceOverSMA.ix[i, 0] < 0.90 and bbp.ix[i, 0] < 0:
                if cashBalance == 0:
                    df_trades.ix[i, 0] = 1000
                    cashBalance += 1000
                    long_trades.append(df_trades.index[i])
                elif cashBalance == -1000:
                    df_trades.ix[i, 0] = 2000
                    cashBalance += 2000
                    long_trades.append(df_trades.index[i])

            elif priceOverSMA.ix[i, 0] > 1.10 and bbp.ix[i, 0] > 0:
                if cashBalance == 0:
                    df_trades.ix[i, 0] = -1000
                    cashBalance -= 1000
                    short_trades.append(df_trades.index[i])
                elif cashBalance == 1000:
                    df_trades.ix[i, 0] = -2000
                    cashBalance -= 2000
                    short_trades.append(df_trades.index[i])

            # Check for MACD crossovers
            elif macd.ix[i - 1, 0] < 0 and macd.ix[i, 0] > 0:
                if cashBalance == 0:
                    df_trades.ix[i, 0] = 1000
                    cashBalance += 1000
                    long_trades.append(df_trades.index[i])
                elif cashBalance == -1000:
                    df_trades.ix[i, 0] = 2000
                    cashBalance += 2000
                    long_trades.append(df_trades.index[i])

            elif macd.ix[i - 1, 0] > 0 and macd.ix[i, 0] < 0:
                if cashBalance == 0:
                    df_trades.ix[i, 0] = -1000
                    cashBalance -= 1000
                    short_trades.append(df_trades.index[i])
                elif cashBalance == 1000:
                    df_trades.ix[i, 0] = -2000
                    cashBalance -= 2000
                    short_trades.append(df_trades.index[i])

        return long_trades, short_trades, df_trades


def author():
    return 'dpanirwala3'


def run_backtest(mss, symbol, start_date, end_date, start_val):
    long, short, df_trades = mss.testPolicy(
        symbol, start_date, end_date, start_val)

    # Calculate portfolio values for the given trades
    portvals = msc.compute_portvals(df_trades, start_val, 9.95, 0.005)
    returns_cumu = (portvals[-1] / portvals[0]) - 1
    daily_returns = portvals / portvals.shift(1) - 1
    daily_returns = daily_returns[1:]
    ADR = daily_returns.mean()
    stddev_returns = daily_returns.std()

    # Calculate benchmark values
    bench_trades = df_trades.copy()
    bench_trades.iloc[:] = 0.0
    bench_trades.iloc[0] = 1000
    bench_port = msc.compute_portvals(
        bench_trades, start_val, 9.95, 0.005)
    bench_cumu = (bench_port[-1] / bench_port[0]) - 1
    bench_DR = bench_port / bench_port.shift(1) - 1
    bench_DR = bench_DR[1:]
    bench_ADR = bench_DR.mean()
    bench_stddev = bench_DR.std()

    return long, short, df_trades, portvals, returns_cumu, ADR, stddev_returns, bench_port, bench_cumu, bench_ADR, bench_stddev


def print_and_plot_results(portvals, bench_portvals, long, short, cum_ret, bench_cum_ret, stddev_returns, bench_stddev_returns, avg_daily_ret, bench_avg_daily_ret, title, start_dt, end_dt):
    print("\n"+title)
    print("\nDate Range: {} to {}".format(start_dt, end_dt))
    print("\nCR of Portfolio: {}".format(cum_ret))
    print("CR of Benchmark: {}".format(bench_cum_ret))
    print("\nStdDev of Portfolio: {}".format(stddev_returns))
    print("StdDev of Benchmark: {}".format(bench_stddev_returns))
    print("\nADR of Portfolio: {}".format(avg_daily_ret))
    print("ADR of Benchmark: {}".format(bench_avg_daily_ret))
    print("\nEnding Portfolio Value of Portfolio: {}".format(portvals[-1]))
    print("Ending Portfolio Value of Benchmark: {}".format(bench_portvals[-1]))

    portval_norm_fund = portvals / portvals.ix[0,]
    bnch_norm_fund = bench_portvals / bench_portvals.ix[0,]

    plt.figure(figsize=(16, 9))

    plt.plot(portval_norm_fund, color='Red', label='Portfolio')
    plt.plot(bnch_norm_fund, color='Purple', label='Benchmark')
    for l in long:
        plt.axvline(x=l, color='blue')
    for s in short:
        plt.axvline(x=s, color='black')
    plt.legend(fontsize=16)
    plt.title(title, fontsize=16)
    plt.xlim(start_dt, end_dt)
    plt.xticks(rotation=25)
    plt.ylabel('Value', fontsize=16)
    plt.xlabel('Dates', fontsize=16)

    plt.savefig('images/' + title + '.png')


def test_code():
    sym = 'JPM'
    start_dt = dt.datetime(2008, 1, 1)
    end_dt = dt.datetime(2009, 12, 31)
    start_val = 100000

    # In-sample
    long, short, df_trades, portvals, cum_ret, avg_daily_ret, stddev_returns, bench_portvals, bench_cum_ret, bench_avg_daily_ret, bench_stddev_returns = run_backtest(
        ManualStrategy(), sym, start_dt, end_dt, start_val)

    print_and_plot_results(portvals, bench_portvals, long, short, cum_ret, bench_cum_ret, stddev_returns, bench_stddev_returns, avg_daily_ret, bench_avg_daily_ret,
                           'Manual Strategy In Sample Portfolio Vs Benchmark', start_dt, end_dt)

    # Out of sample

    start_dt = dt.datetime(2010, 1, 1)
    end_dt = dt.datetime(2011, 12, 31)

    long, short, df_trades, portvals, cum_ret, avg_daily_ret, stddev_returns, bench_portvals, bench_cum_ret, bench_avg_daily_ret, bench_stddev_returns = run_backtest(
        ManualStrategy(), sym, start_dt, end_dt, start_val)

    print_and_plot_results(portvals, bench_portvals, long, short, cum_ret, bench_cum_ret, stddev_returns, bench_stddev_returns, avg_daily_ret, bench_avg_daily_ret,
                           'Manual Strategy Out of Sample Portfolio Vs Benchmark', start_dt, end_dt)


if __name__ == "__main__":
    test_code()
