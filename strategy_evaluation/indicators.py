import pandas as pd
from util import get_data
# import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import datetime as dt

# import matplotlib.gridspec as gridspec


def bollingerbands(sd, ed, symbol,  window_size=20):
    new_sd = sd - dt.timedelta(days=window_size+15)
    df = get_data([symbol], pd.date_range(new_sd, ed))
    price = df[[symbol]]
    temp = price.rolling(window=window_size, min_periods=window_size)
    bbp = (price - temp.mean())/(2 * temp.std())
    return bbp.loc[sd:ed]


def PriceSMA(sd, ed, symbol,  window_size=20):
    new_sd = sd - dt.timedelta(days=window_size+15)
    df = get_data([symbol], pd.date_range(new_sd, ed))
    price = df[symbol]
    crossover = price / \
        price.rolling(window=window_size, min_periods=window_size).mean()
    return crossover.loc[sd:ed]


def MACD(sd, ed, symbol):
    price = get_data([symbol], pd.date_range(sd - dt.timedelta(52), ed))
    price = price[[symbol]].ffill().bfill()
    expMeanA_12, expMeanA_26 = price.ewm(span=12, adjust=False).mean(
    ), price.ewm(span=26, adjust=False).mean()
    macd_raw = expMeanA_12 - expMeanA_26
    macd_signal = macd_raw.ewm(span=9, adjust=False).mean()

    expMeanA_12 = expMeanA_12.truncate(before=sd)
    expMeanA_26 = expMeanA_26.truncate(before=sd)
    macd_raw = macd_raw.truncate(before=sd)
    macd_signal = macd_signal.truncate(before=sd)

    return macd_raw - macd_signal


def author():
    return 'dpanirwala3'


if __name__ == "__main__":
    print("indicators")
