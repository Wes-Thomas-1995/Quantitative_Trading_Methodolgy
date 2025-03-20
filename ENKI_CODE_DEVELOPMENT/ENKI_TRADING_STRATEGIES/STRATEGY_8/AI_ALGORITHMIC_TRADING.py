
from STRATEGY import CONDITIONS

from binance.helpers import round_step_size
from binance.client import Client
from binance.enums import *

from datetime import datetime, timedelta, date, time
from itertools import combinations, product
from time import strftime
import pandas as pd
import numpy as np
import warnings
import os.path
import time
import os
import ta

from dotenv import load_dotenv
import scipy.stats
import torch
import json

from chronos import ChronosPipeline
import importlib.util
import itertools
import warnings
import os.path
import time
import copy
import ast

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







class PREDICTION_LOGIC():
    def __init__(self, PIPELINE, INPUT_PARAMS_USE, INPUT_DATA):
        self.PIPELINE                   = PIPELINE
        self.INPUT_PARAMS_USE           = INPUT_PARAMS_USE
        self.INPUT_DATA                 = INPUT_DATA
        self.LIST                       = self.PREDICTION_LOGIC()

    def PREDICTION_LOGIC(self):

        if self.INPUT_PARAMS_USE[4] == 'TP_AI_1':       
            CONTEXT                             = self.INPUT_DATA['CLOSE'].values[-21:] 
            CONTEXT_TENSOR                      = torch.tensor([np.array(CONTEXT, dtype=np.float32)])  
            FORECAST                            = self.PIPELINE.predict(CONTEXT_TENSOR, 1)  
            LOW, MEDIAN, HIGH                   = np.quantile(FORECAST[0].squeeze().numpy(), [0.1, 0.5, 0.9], axis=0)
            return [MEDIAN]

        else:                                           
            AI_CON                              = self.INPUT_DATA[['OPEN', 'HIGH', 'LOW', 'CLOSE']].values.tolist()
            CONTEXT_                            = AI_CON[-50:]
            CONTEXT_TENSOR_                     = torch.tensor(np.array(CONTEXT_, dtype=np.float32))
            FORECAST_                           = self.PIPELINE.predict(CONTEXT_TENSOR_, 4)
            LOW_, MEDIAN_, HIGH_                = np.quantile(FORECAST_[0].numpy(), [0.1, 0.5, 0.9], axis=0)
            return [np.quantile(FORECAST_[0][:, 3].numpy(), 0.5)]







class PROCESSING():
    def __init__(self, PREDICTIONS, INPUT_DATA, PERIOD=14, RISK_MULTIPLIER=2.0):
        self.PREDICTIONS                        = PREDICTIONS
        self.INPUT_DATA                         = INPUT_DATA
        self.PERIOD                             = PERIOD
        self.RISK_MULTIPLIER                    = RISK_MULTIPLIER
        self.INFORMATION                        = self.PROCESSING()

    def PROCESSING(self):

        STOCH                                   = ta.momentum.StochasticOscillator(self.INPUT_DATA['HIGH'], self.INPUT_DATA['LOW'], self.INPUT_DATA['CLOSE'])
        self.INPUT_DATA['%K']                   = (STOCH.stoch())
        self.INPUT_DATA['%D']                   = (STOCH.stoch_signal())
        self.INPUT_DATA['VOLATILITY']           = (((self.INPUT_DATA['HIGH'] - self.INPUT_DATA['LOW']) / self.INPUT_DATA['CLOSE']) * 100)
        self.INPUT_DATA['AVG_VOLATILITY_20']    = self.INPUT_DATA['VOLATILITY'].rolling(window=20).mean()

        self.INPUT_DATA['High-Low']             = self.INPUT_DATA['HIGH'] - self.INPUT_DATA['LOW']
        self.INPUT_DATA['High-Close']           = abs(self.INPUT_DATA['HIGH'] - self.INPUT_DATA['CLOSE'].shift(1))
        self.INPUT_DATA['Low-Close']            = abs(self.INPUT_DATA['LOW'] - self.INPUT_DATA['CLOSE'].shift(1))
        self.INPUT_DATA['True_Range']           = self.INPUT_DATA[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
        self.INPUT_DATA['ATR']                  = self.INPUT_DATA['True_Range'].rolling(window=self.PERIOD).mean()
        CONTEXT_ROW                             = self.INPUT_DATA[-1:].reset_index(drop=True)
        CLOSE_PRICE                             = CONTEXT_ROW.at[0, 'CLOSE']
        ATR_PRICE                               = CONTEXT_ROW.at[0, 'ATR']
        K                                       = CONTEXT_ROW.at[0, '%K']
        D                                       = CONTEXT_ROW.at[0, '%D']
        AVG_VOL                                 = CONTEXT_ROW.at[0, 'AVG_VOLATILITY_20']
        TP                                      = self.PREDICTIONS[0]
        SIGNAL                                  = 'LONG' if TP > CLOSE_PRICE else 'SHORT'
        SL                                      = (CLOSE_PRICE - (ATR_PRICE * self.RISK_MULTIPLIER)) if SIGNAL == 'LONG' else (CLOSE_PRICE + (ATR_PRICE * self.RISK_MULTIPLIER))
        P_GAIN                                  = round(((TP / CLOSE_PRICE) - 1)*100,2) if SIGNAL == 'LONG' else round((1- (TP / CLOSE_PRICE))*100,2)
        return [SIGNAL, TP, SL, P_GAIN, K, D, AVG_VOL]










class VALIDATION():
    def __init__(self, INFORMATION, INPUT_PARAMS_USE):
        self.INFORMATION                                                = INFORMATION
        self.INPUT_PARAMS_USE                                           = INPUT_PARAMS_USE
        self.INFORMATION                                                = self.VALIDATION()

    def VALIDATION(self):

        SIGNAL_REINFORCED                                                = self.INFORMATION[0] if ((self.INFORMATION[3] >= self.INPUT_PARAMS_USE[3][0]) and (self.INFORMATION[3] <= self.INPUT_PARAMS_USE[3][1])) else 'STATIC'
        if self.INPUT_PARAMS_USE[3][2] == 'STOCH_10':                    SIGNAL_REINFORCED = SIGNAL_REINFORCED if ((self.INFORMATION[4] > 10) and (self.INFORMATION[5] > 10) and (self.INFORMATION[4] < 90) and (self.INFORMATION[5] < 90)) else 'STATIC'
        elif self.INPUT_PARAMS_USE[3][2] == 'STOCH_40':                  SIGNAL_REINFORCED = SIGNAL_REINFORCED if ((self.INFORMATION[4] > 40) and (self.INFORMATION[5] > 40) and (self.INFORMATION[4] < 60) and (self.INFORMATION[5] < 60)) else 'STATIC'
        elif self.INPUT_PARAMS_USE[3][2] == 'AVG_VOL_P_GAIN_3_2':        SIGNAL_REINFORCED = SIGNAL_REINFORCED if (self.INFORMATION[6] > 3 * self.INFORMATION[3]) else 'STATIC'
        return [SIGNAL_REINFORCED, self.INFORMATION[1], self.INFORMATION[2]]










class DEVELOPMENT_CONTROL():
    def __init__(self, INPUT_PARAMS, CLIENT):
        self.CLIENT                             = CLIENT
        self.INPUT_PARAMS                       = INPUT_PARAMS
        self.INFORMATION                        = self.DEVELOPMENT_CONTROL()

    def DEVELOPMENT_CONTROL(self):

        PIPE_TOOL                               = ["amazon/chronos-t5-small", "amazon/chronos-t5-large"]
        DEVICE                                  = "mps" if torch.backends.mps.is_available() else "cpu"
        if self.INPUT_PARAMS[2] == 'SMALL':     PIPELINE = ChronosPipeline.from_pretrained(PIPE_TOOL[0], device_map = DEVICE,  torch_dtype = torch.float32)
        else:                                   PIPELINE = ChronosPipeline.from_pretrained(PIPE_TOOL[1], device_map = DEVICE,  torch_dtype = torch.float32)
        RAW                                     = HISTORICAL_DATA(self.CLIENT, self.INPUT_PARAMS[0], self.INPUT_PARAMS[1])
        PREDICTIONS                             = PREDICTION_LOGIC(PIPELINE, self.INPUT_PARAMS, RAW.DF)
        DATA                                    = PROCESSING(PREDICTIONS.LIST, RAW.DF)
        TRADE                                   = VALIDATION(DATA.INFORMATION, self.INPUT_PARAMS)
        return TRADE.INFORMATION










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
    def __init__(self, INFORMATION, COIN, TIMEFRAME, CONDITIONS, CLIENT, CHECK_POS):
        self.INFORMATION    = INFORMATION
        self.COIN           = COIN
        self.TIMEFRAME      = TIMEFRAME
        self.CONDITIONS     = CONDITIONS
        self.CLIENT         = CLIENT
        self.CHECK_POS      = CHECK_POS
        self.ORDER_OPENED   = self.FINAL_STAGE()



    def FINAL_STAGE(self):

        print(f'\n COIN REVIEWED - {self.COIN}    DECISION - {self.INFORMATION[0]} \n')

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
            DATA                = DEVELOPMENT_CONTROL(INPUT_PARAMS, CLIENT)
            STATUS              = ACTION(DATA.INFORMATION, INPUT_PARAMS[0], INPUT_PARAMS[1], INPUT_PARAMS[5], CLIENT, CHECK.POS_OPEN[0])
            break



    return





if __name__ == '__main__':
    KEY             = 8
    PROJECT_FOLDER  = os.path.expanduser(f'~/CRYPTO_TRADER/ALGO_TRADING_{KEY}/')

    load_dotenv(os.path.join(PROJECT_FOLDER, '.env'))
    FULL_RUN(KEY)


