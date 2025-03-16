
from datetime import datetime, timedelta, date, time
from itertools import combinations
from time import strftime
import pandas as pd
import numpy as np



class STRATEGY_7:

    def __init__(self, DATA_DF):
        self.DATA_DF = DATA_DF
        self.DF = self.STRATEGY()

    def STRATEGY(self):
        TAKE_PROFIT = 0.01
        STOP_LOSS = 0.02
        self.DATA_DF['SIGNAL'] = 'STATIC'
        self.DATA_DF['SIGNAL'] = np.where((self.DATA_DF['EMA_18'].shift(1) <= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] > self.DATA_DF['EMA_24']), 'LONG', np.where((self.DATA_DF['EMA_18'].shift(1) >= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] < self.DATA_DF['EMA_24']), 'SHORT', 'STATIC'))
        self.DATA_DF['TP'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 + TAKE_PROFIT), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 - TAKE_PROFIT), 0))
        self.DATA_DF['SL'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 - STOP_LOSS), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 + STOP_LOSS), 0))
        return self.DATA_DF



class STRATEGY_6:

    def __init__(self, DATA_DF):
        self.DATA_DF = DATA_DF
        self.DF = self.STRATEGY()

    def STRATEGY(self):
        TAKE_PROFIT = 0.01
        STOP_LOSS = 0.02
        self.DATA_DF['SIGNAL'] = 'STATIC'
        self.DATA_DF['SIGNAL'] = np.where((self.DATA_DF['EMA_18'].shift(1) <= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] > self.DATA_DF['EMA_24']), 'LONG', np.where((self.DATA_DF['EMA_18'].shift(1) >= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] < self.DATA_DF['EMA_24']), 'SHORT', 'STATIC'))
        self.DATA_DF['TP'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 + TAKE_PROFIT), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 - TAKE_PROFIT), 0))
        self.DATA_DF['SL'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 - STOP_LOSS), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 + STOP_LOSS), 0))
        return self.DATA_DF



class STRATEGY_1:

    def __init__(self, DATA_DF):
        self.DATA_DF = DATA_DF
        self.DF = self.STRATEGY()

    def STRATEGY(self):
        TAKE_PROFIT = 0.01
        STOP_LOSS = 0.02
        self.DATA_DF['SIGNAL'] = 'STATIC'
        self.DATA_DF['SIGNAL'] = np.where((self.DATA_DF['EMA_18'].shift(1) <= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] > self.DATA_DF['EMA_24']), 'LONG', np.where((self.DATA_DF['EMA_18'].shift(1) >= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] < self.DATA_DF['EMA_24']), 'SHORT', 'STATIC'))
        self.DATA_DF['TP'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 + TAKE_PROFIT), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 - TAKE_PROFIT), 0))
        self.DATA_DF['SL'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 - STOP_LOSS), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 + STOP_LOSS), 0))
        return self.DATA_DF



class STRATEGY_4:

    def __init__(self, DATA_DF):
        self.DATA_DF = DATA_DF
        self.DF = self.STRATEGY()

    def STRATEGY(self):
        TAKE_PROFIT = 0.01
        STOP_LOSS = 0.02
        self.DATA_DF['SIGNAL'] = 'STATIC'
        self.DATA_DF['SIGNAL'] = np.where((self.DATA_DF['EMA_18'].shift(1) <= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] > self.DATA_DF['EMA_24']), 'LONG', np.where((self.DATA_DF['EMA_18'].shift(1) >= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] < self.DATA_DF['EMA_24']), 'SHORT', 'STATIC'))
        self.DATA_DF['TP'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 + TAKE_PROFIT), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 - TAKE_PROFIT), 0))
        self.DATA_DF['SL'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 - STOP_LOSS), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 + STOP_LOSS), 0))
        return self.DATA_DF



class STRATEGY_3:

    def __init__(self, DATA_DF):
        self.DATA_DF = DATA_DF
        self.DF = self.STRATEGY()

    def STRATEGY(self):
        TAKE_PROFIT = 0.01
        STOP_LOSS = 0.02
        self.DATA_DF['SIGNAL'] = 'STATIC'
        self.DATA_DF['SIGNAL'] = np.where((self.DATA_DF['EMA_18'].shift(1) <= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] > self.DATA_DF['EMA_24']), 'LONG', np.where((self.DATA_DF['EMA_18'].shift(1) >= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] < self.DATA_DF['EMA_24']), 'SHORT', 'STATIC'))
        self.DATA_DF['TP'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 + TAKE_PROFIT), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 - TAKE_PROFIT), 0))
        self.DATA_DF['SL'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 - STOP_LOSS), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 + STOP_LOSS), 0))
        return self.DATA_DF



class STRATEGY_2:

    def __init__(self, DATA_DF):
        self.DATA_DF = DATA_DF
        self.DF = self.STRATEGY()

    def STRATEGY(self):
        TAKE_PROFIT = 0.01
        STOP_LOSS = 0.02
        self.DATA_DF['SIGNAL'] = 'STATIC'
        self.DATA_DF['SIGNAL'] = np.where((self.DATA_DF['EMA_18'].shift(1) <= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] > self.DATA_DF['EMA_24']), 'LONG', np.where((self.DATA_DF['EMA_18'].shift(1) >= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] < self.DATA_DF['EMA_24']), 'SHORT', 'STATIC'))
        self.DATA_DF['TP'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 + TAKE_PROFIT), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 - TAKE_PROFIT), 0))
        self.DATA_DF['SL'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 - STOP_LOSS), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 + STOP_LOSS), 0))
        return self.DATA_DF



class STRATEGY_5:

    def __init__(self, DATA_DF):
        self.DATA_DF = DATA_DF
        self.DF = self.STRATEGY()

    def STRATEGY(self):
        TAKE_PROFIT = 0.01
        STOP_LOSS = 0.02
        self.DATA_DF['SIGNAL'] = 'STATIC'
        self.DATA_DF['SIGNAL'] = np.where((self.DATA_DF['EMA_18'].shift(1) <= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] > self.DATA_DF['EMA_24']), 'LONG', np.where((self.DATA_DF['EMA_18'].shift(1) >= self.DATA_DF['EMA_24'].shift(1)) & (self.DATA_DF['EMA_18'] < self.DATA_DF['EMA_24']), 'SHORT', 'STATIC'))
        self.DATA_DF['TP'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 + TAKE_PROFIT), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 - TAKE_PROFIT), 0))
        self.DATA_DF['SL'] = np.where(self.DATA_DF['SIGNAL'] == 'LONG', self.DATA_DF['OPEN'] * (1 - STOP_LOSS), np.where(self.DATA_DF['SIGNAL'] == 'SHORT', self.DATA_DF['OPEN'] * (1 + STOP_LOSS), 0))
        return self.DATA_DF