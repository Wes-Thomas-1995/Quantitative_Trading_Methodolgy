
from STRATEGY import CONDITIONS

from binance.helpers import round_step_size
from binance.client import Client
from binance.enums import *

from datetime import datetime, timedelta, date, time
from itertools import combinations
from dotenv import load_dotenv
from time import strftime
import pandas as pd
import numpy as np
import warnings
import os.path
import time
import os
import ta

warnings.filterwarnings("ignore")





class HISTORICAL_DATA():
    def __init__(self, client, TICKER, TIMEFRAME):
        self.client         = client
        self.TICKER         = TICKER
        self.TIMEFRAME      = TIMEFRAME
        self.DF             = self.GET_DATA()

    def GET_DATA(self):

        INTERVALS   =  {'4H'    : {'PERIODS': 70,       'FUNCTION': self.client.KLINE_INTERVAL_4HOUR},
                        '1D'    : {'PERIODS': 250,      'FUNCTION': self.client.KLINE_INTERVAL_1DAY}}

        PERIODS                 = INTERVALS[self.TIMEFRAME]['PERIODS']
        FUNCTIONS               = INTERVALS[self.TIMEFRAME]['FUNCTION']
        END                     = (datetime.now() + timedelta(days = 1)).date()
        START                   = END - timedelta(days = PERIODS)
        candle                  = (np.asarray(self.client.get_historical_klines(self.TICKER, FUNCTIONS, str(START), str(END))))[:, :6]
        candle                  = pd.DataFrame(candle, columns=['datetime', 'open', 'high', 'low', 'close', 'volume']).astype(float).rename(columns={'datetime':'DATE', 'open':'OPEN', 'high':'HIGH', 'low':'LOW', 'close':'CLOSE', 'volume':'VOLUME'})
        candle.DATE             = pd.to_datetime(candle.DATE, unit='ms')
        return candle










class TECHNICAL_ANALYSIS:
    def __init__(self, DF):
        self.DF = DF
        self.DF = self.TECHNICAL_ANALYSIS()

    def TECHNICAL_ANALYSIS(self):
        PERIODS                                     = [7, 10, 20]
        TA_PARAMS_WIDER_SCOPE                       = [4, 8, 12, 18, 24, 36, 50, 100, 200]
        for k in TA_PARAMS_WIDER_SCOPE:             self.DF[f'EMA_{k}'] = self.DF['CLOSE'].ewm(span=k, adjust=False).mean()

        macd                                        = ta.trend.MACD(self.DF['CLOSE'])
        self.DF['MACD']                             = macd.macd()
        self.DF['MACD_SIGNAL']                      = macd.macd_signal()
        self.DF['MACD_DIFF']                        = macd.macd_diff()

        bollinger                                   = ta.volatility.BollingerBands(self.DF['CLOSE'], window=20, window_dev=2)
        self.DF['BB_UPPER']                         = bollinger.bollinger_hband()
        self.DF['BB_MIDDLE']                        = bollinger.bollinger_mavg()
        self.DF['BB_LOWER']                         = bollinger.bollinger_lband()
        self.DF['BB_WIDTH']                         = self.DF['BB_UPPER'] - self.DF['BB_LOWER']

        adx                                         = ta.trend.ADXIndicator(high=self.DF['HIGH'], low=self.DF['LOW'], close=self.DF['CLOSE'], window=14)
        self.DF['ADX']                              = adx.adx()
        self.DF['ADX_NEG']                          = adx.adx_neg()
        self.DF['ADX_POS']                          = adx.adx_pos()
        self.DF['PSAR']                             = ta.trend.PSARIndicator(high=self.DF['HIGH'], low=self.DF['LOW'], close=self.DF['CLOSE']).psar()
        self.DF['TSI']                              = ta.momentum.TSIIndicator(self.DF['CLOSE']).tsi()

        COLS_TO_MOVE_BACK                           = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        for P in PERIODS:                           self.DF[f'RVOL_{P}'] = self.DF['VOLUME'] / self.DF['VOLUME'].rolling(window=P).mean()
        for COL in COLS_TO_MOVE_BACK:               self.DF[COL]  = self.DF[COL].shift(-1)
        self.DF                                     = self.DF.iloc[200:, :].reset_index(drop=True)

        return self.DF





class APPLY_TP_SL():
    def __init__(self, DF, TP_SL_OPTIONS):
        self.DF                  = DF
        self.TP_SL_OPTIONS       = TP_SL_OPTIONS
        self.DF                  = self.APPLY_TP_SL()

    def APPLY_TP_SL(self):

        self.DF[f'TP_LONG_0']    = self.DF['OPEN'] * (1 + (self.TP_SL_OPTIONS[0]/100))
        self.DF[f'SL_LONG_0']    = self.DF['OPEN'] * (1 - (self.TP_SL_OPTIONS[1]/100))
        self.DF[f'TP_SHORT_0']   = self.DF['OPEN'] * (1 - (self.TP_SL_OPTIONS[0]/100))
        self.DF[f'SL_SHORT_0']   = self.DF['OPEN'] * (1 + (self.TP_SL_OPTIONS[1]/100))
        self.DF['TP']            = np.where((self.DF['SIGNAL'] == "LONG"), self.DF[f'TP_LONG_0'], self.DF[f'TP_SHORT_0'])
        self.DF['SL']            = np.where((self.DF['SIGNAL'] == "LONG"), self.DF[f'SL_LONG_0'], self.DF[f'SL_SHORT_0'])

        return self.DF







class STRATEGY_DEVELOPMENT():
    def __init__(self, DF):
        self.DF                                     = DF
        self.DF                                     = self.STRATEGY_DEVELOPMENT()

    def STRATEGY_DEVELOPMENT(self):

        LIST        = [4, 8, 12, 18, 24, 36, 50, 100, 200]
        PAIR_COMBO  = list(combinations(LIST, 2))
        PERIODS     = [7, 10, 20]

        for I in range(len(PAIR_COMBO)):
            LOWER                                           = f'EMA_{PAIR_COMBO[I][0] if PAIR_COMBO[I][0] < PAIR_COMBO[I][1] else PAIR_COMBO[I][1]}'
            UPPER                                           = f'EMA_{PAIR_COMBO[I][1] if PAIR_COMBO[I][0] < PAIR_COMBO[I][1] else PAIR_COMBO[I][0]}'
            self.DF[f'STRATEGY_{LOWER}__{UPPER}']           = np.where((self.DF[LOWER].shift(1) < self.DF[UPPER].shift(1)) & (self.DF[LOWER] > self.DF[UPPER]), 'LONG', np.where((self.DF[LOWER].shift(1) > self.DF[UPPER].shift(1)) & (self.DF[LOWER] < self.DF[UPPER]), 'SHORT','STATIC'))

        self.DF['STRATEGY_MACD_COMBINED_STRATEGY']           = np.where((self.DF['MACD'] > self.DF['MACD_SIGNAL']) & (self.DF['MACD'] > 0) & (self.DF['MACD_DIFF'] > 0), 'LONG',np.where((self.DF['MACD'] < self.DF['MACD_SIGNAL']) & (self.DF['MACD'] < 0) & (self.DF['MACD_DIFF'] < 0), 'SHORT','STATIC'))
        self.DF['STRATEGY_MACD_SIGNAL_LINE']                 = np.where((self.DF['MACD'] > self.DF['MACD_SIGNAL']), 'LONG',np.where(self.DF['MACD'] < self.DF['MACD_SIGNAL'], 'SHORT', 'STATIC'))
        self.DF['STRATEGY_MACD_ZERO_LINE_STRATEGY']          = np.where(self.DF['MACD'] > 0, 'LONG',np.where(self.DF['MACD'] < 0, 'SHORT', 'STATIC'))
        self.DF['STRATEGY_PSAR']                             = np.where(self.DF['CLOSE'].shift(1) > self.DF['PSAR'], 'LONG',np.where(self.DF['CLOSE'].shift(1) < self.DF['PSAR'], 'SHORT', 'STATIC'))
        self.DF['STRATEGY_TSI']                              = np.where(self.DF['TSI'] > 0, 'LONG',np.where(self.DF['TSI'] < 0, 'SHORT', 'STATIC'))
        self.DF['STRATEGY_ADX']                              = np.where((self.DF['ADX_POS'] > self.DF['ADX_NEG']) & (self.DF['ADX'] > self.DF['ADX'].shift(1)), 'LONG',np.where((self.DF['ADX_NEG'] > self.DF['ADX_POS']) & (self.DF['ADX'] > self.DF['ADX'].shift(1)), 'SHORT','STATIC'))
        self.DF['STRATEGY_BB_MIDDLE_REVERSION']              = np.where((self.DF['CLOSE'].shift(2) < self.DF['BB_MIDDLE']) & (self.DF['CLOSE'].shift(1) > self.DF['BB_MIDDLE']), 'LONG',np.where((self.DF['CLOSE'].shift(2) > self.DF['BB_MIDDLE']) & (self.DF['CLOSE'].shift(1) < self.DF['BB_MIDDLE']), 'SHORT','STATIC'))

        for G in range(len(PERIODS)):                       self.DF[f'STRATEGY_RVOL_{G}'] = np.where(self.DF[f'RVOL_{PERIODS[G]}'] > 1, 'LONG',np.where(self.DF[f'RVOL_{PERIODS[G]}'] < 1, 'SHORT', 'STATIC'))


        return self.DF




class DEVELOPMENT_CONTROL():

    def __init__(self, INPUT_PARAMS, CLIENT):
        self.CLIENT                             = CLIENT
        self.INPUT_PARAMS                       = INPUT_PARAMS
        self.LIST, self.INFORMATION             = self.DEVELOPMENT_CONTROL()


    def DEVELOPMENT_CONTROL(self):

        RAW                                 = HISTORICAL_DATA(self.CLIENT, self.INPUT_PARAMS[0], self.INPUT_PARAMS[1])
        TA_DF                               = TECHNICAL_ANALYSIS(RAW.DF)
        DATA_DF                             = STRATEGY_DEVELOPMENT(TA_DF.DF)
        DATA_DF.DF['SIGNAL']                = DATA_DF.DF[self.INPUT_PARAMS[2]].apply(lambda row: "LONG" if all(val == "LONG" for val in row) else "SHORT" if all(val == "SHORT" for val in row) else "STATIC",axis=1)
        TP_SL_DF                            = APPLY_TP_SL(DATA_DF.DF, self.INPUT_PARAMS[3])
        LIST                                = TP_SL_DF.DF[['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'SIGNAL', 'TP', 'SL']]

        LIST                                = LIST[:-1]
        CURRENT_ROW                         = LIST[-5:].reset_index(drop=True)
        INFORMATION                         = [CURRENT_ROW.at[4, 'SIGNAL'], CURRENT_ROW.at[4, 'TP'], CURRENT_ROW.at[4, 'SL']]


        return CURRENT_ROW, INFORMATION
    



class SIZE_ATTRIBUTES():
    def __init__(self, CLIENT, COIN):
        self.CLIENT                    = CLIENT
        self.COIN                      = COIN
        self.STEP_SIZE, self.TICK_SIZE = self.SIZE_ATTRIBUTES()

    def SIZE_ATTRIBUTES(self):

        INFO = self.CLIENT.futures_exchange_info()
        INFO = INFO['symbols']
        for x in range(len(INFO)):
            if INFO[x]['symbol'] == self.COIN:
                return INFO[x]['quantityPrecision'], INFO[x]['pricePrecision']
            
        return None
    




class GET_BALANCE():
    def __init__(self, CLIENT):
        self.CLIENT             = CLIENT
        self.BALANCE            = self.GET_BALANCE()

    def GET_BALANCE(self):
        
        ACC_BALANCE = self.CLIENT.futures_account_balance()
        for CHECK_BALANCE in ACC_BALANCE:
            if CHECK_BALANCE["asset"] == "USDT":
                BALANCE = round(float(CHECK_BALANCE["balance"]),2)

        return BALANCE
    




class CREATE_ORDER():
    def __init__(self, CLIENT, TICKER, QTY, DIRECTION, LEVERAGE):
        self.CLIENT             = CLIENT
        self.TICKER             = TICKER
        self.QTY                = QTY
        self.DIRECTION          = DIRECTION
        self.LEVERAGE           = LEVERAGE
        self.SIDE               = self.CREATE_ORDER()

    def CREATE_ORDER(self):
        if self.DIRECTION == 'SHORT':       SIDE      = 'SELL'
        else:                               SIDE      = 'BUY'
        
        self.CLIENT.futures_change_leverage(symbol=self.TICKER, leverage=self.LEVERAGE)
        self.CLIENT.futures_create_order(symbol=self.TICKER, side=SIDE, type='MARKET', quantity=self.QTY)

        return SIDE
    




class OPEN_POSITIONS():
    def __init__(self, CLIENT, COIN):
        self.CLIENT                                                     = CLIENT
        self.COIN                                                       = COIN
        self.POS_DF, self.ENTRY_PRICE, self.DIRECTION_EXIST, self.QTY   = self.OPEN_POSITIONS()


    def OPEN_POSITIONS(self):

        QTY, DIRECTION_EXIST, ENTRY_PRICE           = 0, 'NA', 0

        try:
            POS_DF                                  = pd.DataFrame(self.CLIENT.futures_account()['positions'])
            POS_DF                                  = POS_DF[POS_DF['symbol'] == self.COIN].reset_index(drop=True)
            POS_DF['entryPrice']                    = POS_DF['entryPrice'].astype(float)
            POS_DF                                  = POS_DF[POS_DF['entryPrice'] > 0]

            if len(POS_DF) > 0:
                ENTRY_PRICE                         = POS_DF.at[0, 'entryPrice']
                DIRECTION_EXIST                     = float(POS_DF.at[0, 'positionAmt'])
                QTY                                 = abs(float(POS_DF.at[0, 'positionAmt']))
                if DIRECTION_EXIST < 0:             DIRECTION_EXIST = 'SELL'
                else:                               DIRECTION_EXIST = 'BUY'

        except: POS_DF = []
        return POS_DF, ENTRY_PRICE, DIRECTION_EXIST, QTY
    




class CREATE_TP_AND_SL():
    def __init__(self, CLIENT, TICKER, INFORMATION, TICK_SIZE):
        self.CLIENT                 = CLIENT
        self.TICKER                 = TICKER
        self.INFORMATION            = INFORMATION
        self.TICK_SIZE              = TICK_SIZE
        self.SIDE                   = self.CREATE_TP_AND_SL()



    def CREATE_TP_AND_SL(self):


        if self.INFORMATION[0] == 'LONG':   SIDE = 'SELL'
        else:                               SIDE = 'BUY'

        TAKE_PROFIT_PRICE                   = "{:0.0{}f}".format((self.INFORMATION[1]), self.tick_size)
        STOP_LIMIT_PRICE                    = "{:0.0{}f}".format((self.INFORMATION[2]), self.tick_size)
        FIGURE                              = [STOP_LIMIT_PRICE, TAKE_PROFIT_PRICE]
        TYPES                               = ['STOP_MARKET', 'TAKE_PROFIT_MARKET']

        for A in range(2):
            try:                            self.CLIENT.futures_create_order(symbol=self.TICKER, side=SIDE, type=TYPES[A], timeInForce= 'GTE_GTC', stopPrice=FIGURE[A], closePosition='true')
            except:                         pass

        return SIDE





class CLOSE_POSITION():
    def __init__(self, CLIENT, TICKER, QTY, DIRECTION_EXIST):
        self.CLIENT             = CLIENT
        self.TICKER             = TICKER
        self.QTY                = QTY
        self.DIRECTION_EXIST    = DIRECTION_EXIST
        self.DIRECTION          = self.CLOSE_POSITION()

    def CLOSE_POSITION(self):
        if self.DIRECTION_EXIST == 'SELL':      DIRECTION = 'BUY'
        else:                                   DIRECTION = 'SELL'
        self.client.futures_create_order(symbol=self.TICKER, side=DIRECTION, type='MARKET', quantity=self.QTY)
        return DIRECTION
    




class CHECK_IF_POS_OPEN():
    def __init__(self, ROOT):
        self.ROOT               = ROOT
        self.POS_OPEN           = self.CHECK_IF_POS_OPEN()

    def CHECK_IF_POS_OPEN(self):
        
        with open((self.ROOT) + "/CHECK_POS.txt", "r") as f:
            contents = f.readlines()
            POS_OPEN = contents

        return POS_OPEN







class ACTION():
    def __init__(self, LIST_DATA, INFORMATION, COIN, TIMEFRAME, CONDITIONS, CLIENT, CHECK_POS):
        self.LIST_DATA      = LIST_DATA
        self.INFORMATION    = INFORMATION
        self.COIN           = COIN
        self.TIMEFRAME      = TIMEFRAME
        self.CONDITIONS     = CONDITIONS
        self.CLIENT         = CLIENT
        self.CHECK_POS      = CHECK_POS
        self.ORDER_OPENED   = self.FINAL_STAGE()



    def FINAL_STAGE(self):
        print(f'\n COIN REVIEWED - {self.COIN}    DECISION - {self.INFORMATION[0]}')
        print(self.LIST_DATA)

        FOUR_HOUR                   = [0, 4, 8, 12, 16, 20]
        ONE_DAY                     = [0]
        TIME_COR, MOVE_FORWARDS     = False, False

        while TIME_COR == False:

            if self.TIMEFRAME == '4H':
                if ((((datetime.now()).hour + 1) in FOUR_HOUR) or (((datetime.now()).hour) in FOUR_HOUR)):  MOVE_FORWARDS = True

            elif self.TIMEFRAME == '1D':
                if ((((datetime.now()).hour + 1) in ONE_DAY) or (((datetime.now()).hour) in ONE_DAY)):      MOVE_FORWARDS = True

            if ((self.INFORMATION[0] != 'STATIC') and (MOVE_FORWARDS == True)):
                ENTRY_PRICE        = self.OPEN_NEW_POSITION()
                break

        return TIME_COR




    def OPEN_NEW_POSITION(self):
        STEP_INFO               = SIZE_ATTRIBUTES(self.CLIENT, self.COIN)
        CURRENT                 = GET_BALANCE(self.CLIENT)
        BALANCE_TO_TRADE        = CURRENT.BALANCE * self.CONDITIONS[1] * self.CONDITIONS[2]
        PRICE                   = float(self.CLIENT.futures_symbol_ticker(symbol=self.COIN)['price'])
        QTY                     = "{:0.0{}f}".format((BALANCE_TO_TRADE/PRICE), STEP_INFO.STEP_SIZE)

        if ((self.CONDITIONS[0] == 0) and (self.CHECK_POS == 'False')):
            ORDER               = CREATE_ORDER(self.CLIENT, self.COIN, QTY, self.INFORMATION[0], self.CONDITIONS[2])
            time.sleep(10)
            OPEN_POS            = OPEN_POSITIONS(self.CLIENT, self.COIN)
            TP_SL               = CREATE_TP_AND_SL(self.CLIENT, self.COIN, self.INFORMATION, STEP_INFO.TICK_SIZE)

            print(f"\n ORDER OPENED - {self.COIN} - {self.INFORMATION[0]} - Position opened at : {round(OPEN_POS.ENTRY_PRICE, 6)} \n")

        elif ((self.CONDITIONS[0] == 1) or (self.CONDITIONS[0] == 2)):
            if self.CHECK_POS == 'True':   
                OPEN_POS        = OPEN_POSITIONS(self.CLIENT, self.COIN)
                CLOSE           = CLOSE_POSITION(self.CLIENT, self.COIN, OPEN_POS.QTY, OPEN_POS.DIRECTION)
            ORDER               = CREATE_ORDER(self.CLIENT, self.COIN, QTY, self.INFORMATION[0], self.CONDITIONS[2])
            OPEN_POS            = OPEN_POSITIONS(self.CLIENT, self.COIN)

            if (self.CONDITIONS[0] == 2):
                time.sleep(10)
                TP_SL           = CREATE_TP_AND_SL(self.CLIENT, self.COIN, self.INFORMATION, STEP_INFO.TICK_SIZE)

            print(f"\n ORDER OPENED - {self.COIN} - {self.INFORMATION[0]} - Position opened at : {round(OPEN_POS.ENTRY_PRICE, 6)} \n")

        return OPEN_POS.ENTRY_PRICE








def FULL_RUN(KEY):

    API_KEY                     = os.getenv(f"SUB_API_KEY_{KEY}")
    API_SEC                     = os.getenv(f"SUB_API_SEC_{KEY}")
    FOLDER                      = f'/home/ENKIINVESTMENTS/CRYPTO_TRADER/ALGO_TRADING_{KEY}'
    CLIENT                      = Client(API_KEY, API_SEC)
    DONE                        = 'NO'
    INPUT_PARAMS                = CONDITIONS()



    while DONE == 'NO':
        if datetime.now() >= (datetime.now().replace(minute=59, second=40, microsecond=0)) and datetime.now() <= (datetime.now().replace(minute=59, second=59, microsecond=0)):

            CHECK               = CHECK_IF_POS_OPEN(FOLDER)
            DATA                = DEVELOPMENT_CONTROL(INPUT_PARAMS.LIST, CLIENT)
            STATUS              = ACTION(DATA.LIST, DATA.INFORMATION, INPUT_PARAMS.LIST[0], INPUT_PARAMS.LIST[1], INPUT_PARAMS.LIST[4], CLIENT, CHECK.POS_OPEN[0])
            break


    return





if __name__ == '__main__':
    KEY             = 4
    PROJECT_FOLDER  = os.path.expanduser(f'~/CRYPTO_TRADER/ALGO_TRADING_{KEY}/')

    load_dotenv(os.path.join(PROJECT_FOLDER, '.env'))
    FULL_RUN(KEY)


