from datetime import datetime
import os
import requests as requests
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import yfinance as yf
import random
from datetime import datetime, timedelta


class Scraper:
    _tm = str(datetime.today() + timedelta(days=1))
    _1yr = str(datetime.today() - timedelta(days=365))

    def __init__(self):
        self._t = None
        self._names = None
        self._cols = None
    
    def set_cols(self, cols):
        self._cols = cols

    def wrap_df(self, df):
        df = df.drop(['Dividends','Stock Splits'], axis=1)[['Open','High','Low','Close','Adj Close','Volume']]
        return df

    def rand_tick(self):
        self._rand = random.choice(self._names)

    def get_tickers(self):
        return self._names

    def set_tickers(self, tick_str):
        self._names = tick_str.upper().split(' ')
        self._t = yf.Tickers(tick_str)

    def get_ticker_hist(self, tick, period='', start=_1yr, end=_tm, interval='1d'):
        if period:
            return self.wrap_df(self._t.tickers[tick].history(period=period, interval=interval, auto_adjust=False, rounding=True))
        return self.wrap_df(self._t.tickers[tick].history(start=start, end=end, interval=interval, auto_adjust=False, rounding=True))

    def get_ticker_info(self, tick):
        return self._t.tickers[tick].info