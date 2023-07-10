from util import get_data
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd


def bollingerbands(sd, ed, symbol, window_size=20, plot=False):
    new_sd = sd - dt.timedelta(days=window_size+15)
    dates = pd.date_range(new_sd, ed)

    df = get_data([symbol], dates)
    price = df[[symbol]]
    rolling_std = price.rolling(
        window=window_size, min_periods=window_size).std()
    sma = price.rolling(window=window_size, min_periods=window_size).mean()
    upper_band, lower_band, bbp = sma + \
        (2 * rolling_std), sma - (2 * rolling_std), (price - sma)/(2 * rolling_std)

    if plot == True:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, gridspec_kw={
                               'height_ratios': [2, 1]}, figsize=(15, 8))
        ax = ax.flatten()
        fig.suptitle('Bollinger Band - JPM', size=25)

        ax[0].plot(price, label='Price')
        ax[0].plot(sma, label='SMA')
        ax[0].plot(upper_band, label='Upper Band')
        ax[0].plot(lower_band, label='Lower Band')
        ax[0].grid(True)
        ax[0].set_ylabel('Price', fontsize=15)
        ax[0].label_outer()
        ax[0].legend(loc='lower left', fontsize=15)

        ax[1].set_xlim(sd, ed)
        ax[1].set_ylim(-1.5, 1.5)
        ax[1].plot(bbp, label='Bollinger Band Percentage')
        ax[1].set_ylabel('BBP', fontsize=15)
        ax[1].grid(True)

        ax[1].set_xlabel('Date', fontsize=15)
        fig.savefig('images/part2_bollingerbands.png')

    return price, sma, upper_band, lower_band, bbp


def PriceSMA(sd, ed, symbol, window_size=20, plot=False):
    new_sd = sd - dt.timedelta(days=window_size+15)
    dates = pd.date_range(new_sd, ed)
    df = get_data([symbol], dates)
    price = df['JPM']
    sma = price.rolling(window=window_size, min_periods=window_size).mean()

    crossover = price/sma

    if plot == True:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False,
                               gridspec_kw={'height_ratios': [2, 1]},
                               figsize=(15, 8))

        ax = ax.flatten()
        fig.suptitle('Price/SMA - JPM', size=20)

        ax[0].plot(price, label='Price')
        ax[0].plot(sma, label='SMA')
        ax[0].grid(True)
        ax[0].set_ylabel('Price', fontsize=15)
        ax[0].label_outer()
        ax[0].legend(loc='lower left')

        ax[1].set_xlim(sd, ed)
        ax[1].set_ylim(0.6, 1.30)
        ax[1].plot(crossover, label='Price/SMA')
        ax[1].set_ylabel('images/part2_sma', fontsize=15)
        d = price.index
        y = 1.05

        ax[1].fill_between(
            d, crossover, y, where=crossover >= y, facecolor='pink')

        y2 = 0.95
        ax[1].fill_between(
            d, crossover, y2, where=crossover <= y2, facecolor='pink')

        ax[1].grid(True)

        ax[1].set_xlabel('Date', fontsize=15)
        fig.savefig('images/part2_price_by_sma.png')

    return price, sma, crossover


def ema(sd, ed, symbol, window_size=20, plot=False):
    price = get_data([symbol], pd.date_range(
        sd - dt.timedelta(window_size * 2), ed))
    price = price[[symbol]].ffill().bfill()
    df_ema = price.ewm(
        span=window_size, adjust=False).mean().truncate(before=sd)
    price = price.truncate(before=sd)
    norm_df_price, norm_df_ema = price[symbol] / \
        price[symbol][0], df_ema[symbol] / df_ema[symbol][0]

    if plot == True:
        plt.figure(figsize=(14, 8))

        plt.title("{} window days EMA".format(window_size))
        plt.xlabel("Date")
        plt.ylabel("Normalized Pirce")
        plt.xticks(rotation=30)
        plt.grid()
        plt.plot(norm_df_price, label="normalized price", color="blue")
        plt.plot(norm_df_ema, label="{} window days EMA".format(
            window_size), color="red")
        plt.legend()
        plt.savefig("images/part2_ema.png", bbox_inches='tight')
        plt.clf()

    return norm_df_ema


def macd(sd, ed, symbol, plot=False):
    price = get_data([symbol], pd.date_range(sd - dt.timedelta(52), ed))
    price = price[[symbol]].ffill().bfill()

    expMeanA_12, expMeanA_26 = price.ewm(span=12, adjust=False).mean(
    ), price.ewm(span=26, adjust=False).mean()
    macd_raw = expMeanA_12 - expMeanA_26
    macd_signal = macd_raw.ewm(span=9, adjust=False).mean()

    price = price.truncate(before=sd)
    expMeanA_12 = expMeanA_12.truncate(before=sd)
    expMeanA_26 = expMeanA_26.truncate(before=sd)
    macd_raw = macd_raw.truncate(before=sd)
    macd_signal = macd_signal.truncate(before=sd)

    if plot == True:

        fig = plt.figure(figsize=(14, 8))
        plt.suptitle("MACD")
        plt.xlabel("Date")
        plt.ylabel('normalized price')

        norm_expMeanA_12 = expMeanA_12[symbol] / expMeanA_12[symbol][0]
        norm_expMeanA_26 = expMeanA_26[symbol] / expMeanA_26[symbol][0]
        norm_df_price = price[symbol] / price[symbol][0]

        ax1 = plt.subplot(211)
        ax1.plot(norm_expMeanA_12, label="12 days EMA", color="orange")
        ax1.plot(norm_expMeanA_26, label="26 days EMA", color="red")
        ax1.plot(norm_df_price, label="normalized price", color="blue")
        ax1.legend()
        plt.xlabel("Date")
        plt.ylabel('Normalized price')
        ax1.grid()

        ax2 = plt.subplot(212)
        ax2.plot(macd_raw, label="MACD", color="orange")
        ax2.plot(macd_signal, label="MACD Signal", color="red")
        ax2.grid()
        plt.xlabel("Date")
        ax2.legend()

        fig.autofmt_xdate()

        plt.savefig("images/part2_macd.png", bbox_inches='tight')
        # plt.show()
        plt.clf()

    return macd_raw, macd_signal


def tsi(sd, ed, symbol, plot=False):
    price = get_data([symbol], pd.date_range(sd - dt.timedelta(50), ed))
    price = price[[symbol]].ffill().bfill()
    diff = price - price.shift(1)
    expMeanA_25 = diff.ewm(span=25, adjust=False).mean()
    expMeanA_13 = expMeanA_25.ewm(span=13, adjust=False).mean()
    abs_diff = abs(diff)
    abs_expMeanA_25 = abs_diff.ewm(span=25, adjust=False).mean()
    abs_expMeanA_13 = abs_expMeanA_25.ewm(span=13, adjust=False).mean()

    df_tsi = expMeanA_13 / abs_expMeanA_13

    df_tsi = df_tsi.truncate(before=sd)

    if plot == True:
        fig = plt.figure(figsize=(14, 8))
        plt.suptitle("TSI")
        plt.xlabel("Date")
        plt.ylabel('Ratio')

        norm_df_price = price[symbol] / price[symbol][0]

        ax1 = plt.subplot(211)
        ax1.plot(norm_df_price, label="normalized price", color="blue")
        ax1.legend()
        plt.xlabel("Date")
        plt.ylabel('Normalized price')
        ax1.grid()

        ax2 = plt.subplot(212)
        ax2.plot(df_tsi, label="TSI", color="orange")
        ax2.grid()
        plt.xlabel("Date")
        ax2.legend()

        fig.autofmt_xdate()

        plt.savefig("images/part2_tsi.png", bbox_inches='tight')
        plt.clf()

    return df_tsi


def run():
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    symbol = 'JPM'

    expMeanA_data = ema(sd, ed, symbol, plot=True, window_size=20)
    macd_data = macd(sd, ed, symbol, plot=True)
    tsi_data = tsi(sd, ed, symbol, plot=True)


def author():
    return 'dpanirwala3'


if __name__ == "__main__":
    run()
