from util import get_data, plot_data
import datetime as dt
import pandas as pd
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
import indicators
import TheoreticallyOptimalStrategy as tos


def calculateBenchmark(sd, ed, sv):
    df_trades = get_data(['SPY'], pd.date_range(sd, ed))
    df_trades = df_trades.rename(
        columns={'SPY': 'JPM'}).astype({'JPM': 'int32'})
    df_trades[:] = 0
    df_trades.loc[df_trades.index[0]] = 1000
    portvals = compute_portvals(df_trades, sv, commission=9.95, impact=0.005)
    return portvals


def printStats(benchmark, theoretical):
    benchmark, theoretical = benchmark['value'], theoretical['value']
    CUR_bench = benchmark[-1] / benchmark[0] - 1
    cr_the = theoretical[-1] / theoretical[0] - 1
    dr_ben = (benchmark / benchmark.shift(1) - 1).iloc[1:]
    dr_the = (theoretical / theoretical.shift(1) - 1).iloc[1:]
    sddr_ben = dr_ben.std()
    sddr_the = dr_the.std()
    adr_ben = dr_ben.mean()
    adr_the = dr_the.mean()

    print("")
    print("Theoretical Strategy")
    print("Cumulative returns: " + str(cr_the))
    print("Standard Deviation of daily returns: " + str(sddr_the))
    print("Mean of daily returns: " + str(adr_the))
    print("")
    print("Benchmark Strategy")
    print("Cumulative return: " + str(CUR_bench))
    print("Stand Deviation of daily returns: " + str(sddr_ben))
    print("Mean of daily returns: " + str(adr_ben))
    print("")


def plotGraphs(benchmark_portfolio_values, theoretical_portvals):

    # normalize
    benchmark_portfolio_values['value'] = benchmark_portfolio_values['value'] / \
        benchmark_portfolio_values['value'][0]
    theoretical_portvals['value'] = theoretical_portvals['value'] / \
        theoretical_portvals['value'][0]

    plt.figure(figsize=(14, 8))
    plt.title("Theoretically Optimal Strategy")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.xticks(rotation=30)
    plt.grid()
    plt.plot(benchmark_portfolio_values, label="benchmark", color="purple")
    plt.plot(theoretical_portvals, label="theoretical", color="red")
    plt.legend()
    plt.savefig("images/theoretical.png", bbox_inches='tight')
    plt.clf()


def run():
    sv = 100000
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    symbol = 'JPM'

    # Run Indicators
    ema_20 = indicators.ema(sd, ed, symbol, window_size=20, plot=True)
    macd_raw, macd_signal = indicators.macd(sd, ed, symbol, plot=True)
    tsi = indicators.tsi(sd, ed, symbol, plot=True)
    bbp_price, bbp_sma, upper_band, lower_band, bbp = indicators.bollingerbands(
        sd, ed, symbol, window_size=20, plot=True)

    psma_price, psma_sma, crossover = indicators.PriceSMA(
        sd, ed, symbol, window_size=20, plot=True)
    pricebysma = crossover.loc[sd:ed]

    # Run Theoretical Strategy
    df_trades = tos.testPolicy(
        symbol=symbol, sd=sd, ed=ed, sv=sv)
    theoretical_portfolio_values = compute_portvals(
        df_trades, sv, commission=0, impact=0)

    # get benchmark performance
    benchmark_portfolio_values = calculateBenchmark(sd, ed, sv)

    # get stats
    printStats(benchmark_portfolio_values, theoretical_portfolio_values)

    # plot graph
    plotGraphs(benchmark_portfolio_values, theoretical_portfolio_values)


def author():
    return 'dpanirwala3'


if __name__ == "__main__":
    run()
