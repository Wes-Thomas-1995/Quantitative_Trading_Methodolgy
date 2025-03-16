from dotenv import load_dotenv
from itertools import combinations
from binance.helpers import round_step_size
from binance.client import Client
from binance.enums import *
from time import strftime
from datetime import datetime, timedelta, date, time

import pandas as pd
import numpy as np
import warnings
import os.path
import time
import os
import ta

warnings.filterwarnings("ignore")










class GET_BALANCE():
    def __init__(self, client):
        self.client             = client
        self.BALANCE            = self.GET_BALANCE()

    def GET_BALANCE(self):
        acc_balance = self.client.futures_account_balance()
        for check_balance in acc_balance:
            if check_balance["asset"] == "USDT":
                BALANCE = round(float(check_balance["balance"]),2)

        return BALANCE




class OPEN_POSITIONS():

    def __init__(self, client, COIN):
        self.client                                                     = client
        self.COIN                                                       = COIN
        self.POS_DF, self.ENTRY_PRICE, self.DIRECTION_EXIST, self.QTY,  = self.OPEN_POSITIONS()


    def OPEN_POSITIONS(self):

        QTY, DIRECTION_EXIST, ENTRY_PRICE     = 0, 'NA', 0

        try:

            POS_DF                                = pd.DataFrame(self.client.futures_account()['positions'])
            POS_DF                                = POS_DF[POS_DF['symbol'] == self.COIN]
            POS_DF['entryPrice']                  = POS_DF['entryPrice'].astype(float)
            POS_DF                                = POS_DF[POS_DF['entryPrice'] > 0]

            if len(POS_DF) > 0:
                POS_DF               = POS_DF[POS_DF['symbol'] == self.COIN].reset_index()
                ENTRY_PRICE          = POS_DF.at[0, 'entryPrice']
                DIRECTION_EXIST      = float(POS_DF.at[0, 'positionAmt'])
                QTY                  = abs(float(POS_DF.at[0, 'positionAmt']))

                if DIRECTION_EXIST < 0:
                    DIRECTION_EXIST = 'SELL'
                else:
                    DIRECTION_EXIST = 'BUY'

        except:
            print('UNABLE -- Unable to get POS_DF - Assuming no positions open')
            POS_DF = []

        return POS_DF, ENTRY_PRICE, DIRECTION_EXIST, QTY



class CLOSE_POSITION():
    def __init__(self, client, TICKER, QTY, DIRECTION_EXIST):
        self.client             = client
        self.TICKER             = TICKER
        self.QTY                = QTY
        self.DIRECTION_EXIST    = DIRECTION_EXIST
        self.CLOSE_POSITION()

    def CLOSE_POSITION(self):
        if self.DIRECTION_EXIST == 'SELL':      DIRECTION = 'BUY'
        else:                                   DIRECTION = 'SELL'
        self.client.futures_create_order(symbol=self.TICKER, side=DIRECTION, type='MARKET', quantity=self.QTY)

        return




class CREATE_ORDER():
    def __init__(self, client, TICKER, QTY, DIRECTION, LEVERAGE):
        self.client             = client
        self.TICKER             = TICKER
        self.QTY                = QTY
        self.DIRECTION          = DIRECTION
        self.LEVERAGE           = LEVERAGE
        self.CREATE_ORDER()

    def CREATE_ORDER(self):
        if self.DIRECTION == 'SHORT':       SIDE      = 'SELL'
        else:                               SIDE      = 'BUY'
        self.client.futures_change_leverage(symbol=self.TICKER, leverage=self.LEVERAGE)
        self.client.futures_create_order(symbol=self.TICKER, side=SIDE, type='MARKET', quantity=self.QTY)

        return




class CREATE_TP_AND_SL():
    def __init__(self, client, TICKER, ENTRY_PRICE, DIRECTION, TAKE_PROFIT, STOP_LOSS, tick_size, TAKE_PROFIT_INC_LT, STOP_LOSS_INC_LT):
        self.client               = client
        self.TICKER               = TICKER
        self.ENTRY_PRICE          = ENTRY_PRICE
        self.DIRECTION            = DIRECTION
        self.TAKE_PROFIT          = TAKE_PROFIT
        self.STOP_LOSS            = STOP_LOSS
        self.tick_size            = tick_size
        self.TAKE_PROFIT_INC_LT   = TAKE_PROFIT_INC_LT
        self.STOP_LOSS_INC_LT     = STOP_LOSS_INC_LT
        self.CREATE_TP_AND_SL()


    def CREATE_TP_AND_SL(self):
        if self.DIRECTION == 'BUY':
            SIDE                = 'SELL'
            TAKE_PROFIT_PRICE   = self.ENTRY_PRICE * (1 + (self.TAKE_PROFIT/100))
            STOP_LIMIT_PRICE    = self.ENTRY_PRICE * (1 - (self.STOP_LOSS/100))

        else:
            SIDE = 'BUY'
            TAKE_PROFIT_PRICE   = self.ENTRY_PRICE * (1 - (self.TAKE_PROFIT/100))
            STOP_LIMIT_PRICE    = self.ENTRY_PRICE * (1 + (self.STOP_LOSS/100))
        TAKE_PROFIT_PRICE       = "{:0.0{}f}".format((TAKE_PROFIT_PRICE), self.tick_size)
        STOP_LIMIT_PRICE        = "{:0.0{}f}".format((STOP_LIMIT_PRICE), self.tick_size)

        if  self.STOP_LOSS_INC_LT== 'YES':
            try:
                try:            self.client.futures_create_order(symbol=self.TICKER, side=SIDE, type='STOP_MARKET', timeInForce= 'GTE_GTC', stopPrice=STOP_LIMIT_PRICE, closePosition='true')
                except:
                                time.sleep(1)
                                self.client.futures_create_order(symbol=self.TICKER, side=SIDE, type='STOP_MARKET', timeInForce= 'GTE_GTC', stopPrice=STOP_LIMIT_PRICE, closePosition='true')
            except:             pass

        if  self.TAKE_PROFIT_INC_LT== 'YES':
            try:
                try:            self.client.futures_create_order(symbol=self.TICKER, side=SIDE, type='TAKE_PROFIT_MARKET', timeInForce= 'GTE_GTC', stopPrice=TAKE_PROFIT_PRICE, closePosition='true')
                except:
                                time.sleep(1)
                                self.client.futures_create_order(symbol=self.TICKER, side=SIDE, type='TAKE_PROFIT_MARKET', timeInForce= 'GTE_GTC', stopPrice=TAKE_PROFIT_PRICE, closePosition='true')
            except:             pass
        return




class SIZE_2():
    def __init__(self, client, COIN):
        self.client                    = client
        self.COIN                      = COIN
        self.STEP_SIZE, self.TICK_SIZE = self.SIZE_2()

    def SIZE_2(self):

        info = self.client.futures_exchange_info()
        info = info['symbols']
        for x in range(len(info)):
            if info[x]['symbol'] == self.COIN:

                return info[x]['quantityPrecision'], info[x]['pricePrecision']
        return None




class CHECK_IF_POS_OPEN():

    def __init__(self, ROOT):
        self.ROOT              = ROOT
        self.POS_OPEN           = self.CHECK_IF_POS_OPEN()

    def CHECK_IF_POS_OPEN(self):

        POS_OPEN = False

        with open((self.ROOT) + "/CHECK_POS.txt", "r") as f:
            contents = f.readlines()
            POS_OPEN = contents

        return POS_OPEN










class HISTORICAL_DATA():
    def __init__(self, client, TICKER, INTERVAL, PERIOD):
        self.client     = client
        self.TICKER     = TICKER
        self.INTERVAL   = INTERVAL
        self.PERIOD     = PERIOD
        self.DF       = self.get_data()

    def get_data(self):
        END                     = (datetime.now() + timedelta(days = 1)).date()
        START                   = END - timedelta(days = self.PERIOD)
        intervals               = { '1m' : self.client.KLINE_INTERVAL_1MINUTE,
                                    '5m' : self.client.KLINE_INTERVAL_5MINUTE,
                                    '15m': self.client.KLINE_INTERVAL_15MINUTE,
                                    '30m': self.client.KLINE_INTERVAL_30MINUTE,
                                    '1h' : self.client.KLINE_INTERVAL_1HOUR,
                                    '2h' : self.client.KLINE_INTERVAL_2HOUR,
                                    '4h' : self.client.KLINE_INTERVAL_4HOUR,
                                    '6h' : self.client.KLINE_INTERVAL_6HOUR,
                                    '8h' : self.client.KLINE_INTERVAL_8HOUR,
                                    '12h': self.client.KLINE_INTERVAL_12HOUR,
                                    '1d' : self.client.KLINE_INTERVAL_1DAY,
                                    '3d' : self.client.KLINE_INTERVAL_3DAY,
                                    '1w' : self.client.KLINE_INTERVAL_1WEEK,
                                    '1M' : self.client.KLINE_INTERVAL_1MONTH}

        candle                  = np.asarray(self.client.get_historical_klines(self.TICKER, intervals.get(self.INTERVAL), str(START), str(END)))
        candle                  = candle[:, :6]
        candle                  = pd.DataFrame(candle, columns=['datetime', 'open', 'high', 'low', 'close', 'volume']).astype(float).rename(columns={'datetime':'DATE', 'open':'OPEN', 'high':'HIGH', 'low':'LOW', 'close':'CLOSE', 'volume':'VOLUME'})
        candle.DATE             = pd.to_datetime(candle.DATE, unit='ms')
        return candle







class ANALYSIS():

    def __init__(self, DF, INPUT_PARAMS,ATR_PERIOD=10, ATR_MULTIPLIER=0.35):
        self.DF                                 = DF
        self.ATR_PERIOD                         = ATR_PERIOD
        self.ATR_MULTIPLIER                     = ATR_MULTIPLIER
        self.INPUT_PARAMS                       = INPUT_PARAMS
        self.DF                                 = self.ANALYSIS()


    def ANALYSIS(self):

        EMA_LIST, EMA_NBRS                  = ['EMA_100', 'EMA_200'], [100, 200]
        ENTRY_PRICE                         = self.DF['OPEN']

        STATIC_TP = 0.75/100

        for k in range(len(EMA_NBRS)):  self.DF[EMA_LIST[k]]                        = self.DF['CLOSE'].ewm(span=EMA_NBRS[k], adjust=False).mean()
        for k in range(len(EMA_NBRS)):  self.DF[EMA_LIST[k]]                        = self.DF[EMA_LIST[k]].shift(1)

        STOCH                               = ta.momentum.StochasticOscillator(self.DF['HIGH'], self.DF['LOW'], self.DF['CLOSE'])
        self.DF['RSI']                      = ta.momentum.RSIIndicator(self.DF['CLOSE'], window=14).rsi()
        self.DF['%K']                       = STOCH.stoch()
        self.DF['%D']                       = STOCH.stoch_signal()

        self.DF['RSI']                      = self.DF['RSI'].shift(1)
        self.DF['%K']                       = self.DF['%K'].shift(1)
        self.DF['%D']                       = self.DF['%D'].shift(1)
        self.DF                             = self.DF.iloc[200:,:].reset_index().iloc[:,1:]


        self.DF['TRUE_DATE']                = (self.DF['DATE'].dt.date).astype("string")
        self.DF['TIME']                     = (self.DF['DATE'].dt.time).astype("string")
        self.DF['STR_TIMEZONE']             = self.DF['TRUE_DATE'] + ' ' + self.DF['TIME']


        self.DF['DIR']                      = np.where(self.DF['OPEN']<=self.DF['CLOSE'], "LONG", "SHORT")
        self.DF['GAIN_UP']                  = (self.DF['HIGH']/self.DF['OPEN']) - 1
        self.DF['GAIN_DOWN']                = (self.DF['OPEN']/self.DF['LOW']) - 1

        self.DF['PREV_DIR']                 = self.DF['DIR'].shift(1)
        self.DF['PREV_GAIN_UP']             = self.DF['GAIN_UP'].shift(1)
        self.DF['PREV_GAIN_DOWN']           = self.DF['GAIN_DOWN'].shift(1)


        self.DF['ATR']                      = ta.volatility.AverageTrueRange(high=self.DF['HIGH'], low=self.DF['LOW'], close=self.DF['CLOSE'], window=self.ATR_PERIOD).average_true_range() * self.ATR_MULTIPLIER
        self.DF['ATR']                      = self.DF['ATR'].shift(1)
        self.DF['ATR_CONF_3']               = np.where((abs((self.DF['ATR']/self.DF['OPEN'])) >= (STATIC_TP * 3)), True, False)

        self.DF['RSI_FLAG_30_70']           = (self.DF['RSI'] > 30) & (self.DF['RSI'] < 70)
        self.DF['STOCH_FLAG_10_90']         = (self.DF['%K'] > 10) & (self.DF['%K'] < 90) & (self.DF['%D'] > 10) & (self.DF['%D'] < 90)

        self.DF['SIGNAL']                   = self.DF['PREV_DIR']
        self.DF['EMA_100_200']              = np.where((self.DF['EMA_100'] > self.DF['EMA_200']), 'UP', np.where((self.DF['EMA_100'] < self.DF['EMA_200']), 'DOWN', 'STATIC'))
        self.DF['RSI_FLAG']                 = ((self.DF['RSI'] > 40) & (self.DF['RSI'] < 70))
        self.DF['TEST']                     = ((self.DF['SIGNAL'] == "DOWN") & (self.DF['PREV_GAIN_DOWN'] < 0.03))  | ((self.DF['SIGNAL'] == "UP") & (self.DF['PREV_GAIN_UP'] < 0.03))
        self.DF['OTHER TEST']               = ((self.DF['PREV_GAIN_UP'] > 0.05)  |  (self.DF['PREV_GAIN_DOWN'] > 0.05))


        self.DF['CON_1']                    = ((self.DF['EMA_100_200'] == "UP") & (self.DF['RSI_FLAG'] == False))
        self.DF['CON_2']                    = ((self.DF['TEST'] == True) & (self.DF['RSI_FLAG'] == False))
        self.DF['CON_3']                    = ((self.DF['TEST'] == False) & (self.DF['RSI_FLAG_30_70'] == False))
        self.DF['CON_4']                    = ((self.DF['OTHER TEST'] == True) & (self.DF['ATR_CONF_3'] == False) & (self.DF['STOCH_FLAG_10_90'] == True))

        if len(self.INPUT_PARAMS) == 4:      self.DF['CONFIRMATION']         =  (self.DF['CON_' + str(self.INPUT_PARAMS[0])]) | (self.DF['CON_' + str(self.INPUT_PARAMS[1])]) | (self.DF['CON_' + str(self.INPUT_PARAMS[2])]) | (self.DF['CON_' + str(self.INPUT_PARAMS[3])])
        elif len(self.INPUT_PARAMS) == 3:    self.DF['CONFIRMATION']         =  (self.DF['CON_' + str(self.INPUT_PARAMS[0])]) | (self.DF['CON_' + str(self.INPUT_PARAMS[1])]) | (self.DF['CON_' + str(self.INPUT_PARAMS[2])])
        elif len(self.INPUT_PARAMS) == 2:    self.DF['CONFIRMATION']         =  (self.DF['CON_' + str(self.INPUT_PARAMS[0])]) | (self.DF['CON_' + str(self.INPUT_PARAMS[1])])
        elif len(self.INPUT_PARAMS) == 1:    self.DF['CONFIRMATION']         =  (self.DF['CON_' + str(self.INPUT_PARAMS[0])])
        self.DF['SIGNAL']                   = np.where((self.DF['CONFIRMATION']), self.DF['PREV_DIR'], "STATIC")

        KEEP_LIST                           = ['DATE', 'OPEN', 'CLOSE', 'SIGNAL']
        self.DF                             = self.DF[KEEP_LIST]


        return self.DF







class DATA_RUN():

    def __init__(self, STRATEGY_1_CONDITIONS, client):
        self.STRATEGY_1_CONDITIONS              = STRATEGY_1_CONDITIONS
        self.client                             = client
        self.ACTIONS, self.STRATEGY             = self.DATA_RUN()


    def DATA_RUN(self):

        STRATEGY, ACTIONS                       = [], []

        for a in range(len(self.STRATEGY_1_CONDITIONS['COIN'])):
            STRATEGY.append(a)
            ACTIONS.append(a)

            DF_RAW                              = HISTORICAL_DATA(self.client, self.STRATEGY_1_CONDITIONS['COIN'][a], '1d', 210)
            STRATEGY_OUTPUT                     = ANALYSIS(DF_RAW.DF, self.STRATEGY_1_CONDITIONS['INPUT_PARAMS'][a])

            STRATEGY[a]                         = STRATEGY_OUTPUT.DF.copy()
            ACTIONS[a]                          = STRATEGY[a].at[len(STRATEGY[a])-1, 'SIGNAL']


        return ACTIONS, STRATEGY







class ACTION_M1():
    def __init__(self, STRATEGY_1_CONDITIONS_DAILY, ACTIONS, STRATEGY, client):
        self.client                                                                         = client
        self.STRATEGY_1_CONDITIONS_DAILY                                                    = STRATEGY_1_CONDITIONS_DAILY
        self.ACTIONS                                                                        = ACTIONS
        self.STRATEGY                                                                       = STRATEGY
        self.ORDER_OPENED                                                                   = self.FINAL_STAGE()

    def FINAL_STAGE(self):

        def OPEN_NEW_POSITION(client, STRATEGY_1_CONDITIONS, DIRECTION, a):
            STEP_INFO                                                                       = SIZE_2(client, STRATEGY_1_CONDITIONS['COIN'][a])
            CURRENT                                                                         = GET_BALANCE(client)
            BALANCE_TO_TRADE                                                                = ((CURRENT.BALANCE)*0.50) * 6
            PRICE                                                                           = float(client.futures_symbol_ticker(symbol=STRATEGY_1_CONDITIONS['COIN'][a])['price'])
            QTY                                                                             = "{:0.0{}f}".format((BALANCE_TO_TRADE/PRICE), STEP_INFO.STEP_SIZE)
            CREATE_ORDER(client, STRATEGY_1_CONDITIONS['COIN'][a], QTY, DIRECTION, 5)
            DF_RAW_2                                                                        = HISTORICAL_DATA(client, STRATEGY_1_CONDITIONS['COIN'][a], '1h', 1)
            PRICE_ENTRY                                                                     = DF_RAW_2.DF.at[len(DF_RAW_2.DF)-1, 'OPEN']

            time.sleep(10)
            OPEN_POS = OPEN_POSITIONS(client, STRATEGY_1_CONDITIONS['COIN'][a])
            CREATE_TP_AND_SL(client, STRATEGY_1_CONDITIONS['COIN'][a], PRICE_ENTRY, OPEN_POS.DIRECTION_EXIST, STRATEGY_1_CONDITIONS['TAKE_PROFIT'][a],  STRATEGY_1_CONDITIONS['STOP_LOSS'][a], STEP_INFO.TICK_SIZE, 'YES', 'YES')
            return OPEN_POS.ENTRY_PRICE



        TIME_COR, REF_TIME  = 'NO', (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).time()

        for a in range(len(self.ACTIONS)):
                print("\n")
                print('COIN REVIEWED - ' + self.STRATEGY_1_CONDITIONS_DAILY['COIN'][a] + '    DECISION - ' + self.ACTIONS[a])
                print(self.STRATEGY[a])



        for a in range(len(self.ACTIONS)):
            if TIME_COR == "YES": break
            if self.ACTIONS[a] != 'STATIC':
                while TIME_COR == 'NO':
                    if ((datetime.now().replace(minute=0, second=0, microsecond=0)).time() == REF_TIME):

                        ENTRY_PRICE                                             = OPEN_NEW_POSITION(self.client, self.STRATEGY_1_CONDITIONS_DAILY, self.ACTIONS[a], a)
                        TIME_COR                                                = "YES"

                        print("\n")
                        print('ORDER OPENED - ' + self.STRATEGY_1_CONDITIONS_DAILY['COIN'][a] + ' - ' + self.ACTIONS[a] + ' - Position opened at : ' + str(round(ENTRY_PRICE, 6)))
                        print("\n")

                    else:                                                       time.sleep(0.1)



        return TIME_COR







def FULL_RUN():

    STRATEGY_1_CONDITIONS       = { 'COIN'            :  {0      : 'XRPUSDT',
                                                          1      : 'DOGEUSDT',
                                                          2      : 'ADAUSDT'},

                                    'INPUT_PARAMS'    :  {0      : [2, 3, 4],
                                                          1      : [1],
                                                          2      : [1, 3]},

                                    'TAKE_PROFIT'     :  {0      : 0.75,
                                                          1      : 0.75,
                                                          2      : 0.75},

                                    'STOP_LOSS'       :  {0      : 18,
                                                          1      : 18,
                                                          2      : 18}}






    client                      = Client(os.getenv("SUB_API_KEY_2"), os.getenv("SUB_API_SEC_2"))
    DONE                        = 'NO'


    while DONE == 'NO':
        while DONE == 'NO':
            if datetime.now() >= (datetime.now().replace(minute=58, second=0, microsecond=0)) and datetime.now() <= (datetime.now().replace(minute=59, second=4, microsecond=0)):
                CHECK                       = CHECK_IF_POS_OPEN((r'/home/ENKIINVESTMENTS/CRYPTO_TRADER/ALGO_TRADING_2'))
                break


        while DONE == 'NO':
            if datetime.now() >= (datetime.now().replace(minute=59, second=40, microsecond=0)) and datetime.now() <= (datetime.now().replace(minute=59, second=59, microsecond=0)):
                if CHECK.POS_OPEN[0] == 'False':

                        COIN                    = DATA_RUN(STRATEGY_1_CONDITIONS, client)
                        STATUS                  = ACTION_M1(STRATEGY_1_CONDITIONS, COIN.ACTIONS, COIN.STRATEGY, client)

                else:                       print('POSITION CURRENTLY OPEN')
                break

        break

    return











if __name__ == '__main__':
    project_folder = os.path.expanduser('~/CRYPTO_TRADER/ALGO_TRADING_2/')
    load_dotenv(os.path.join(project_folder, '.env'))
    FULL_RUN()


