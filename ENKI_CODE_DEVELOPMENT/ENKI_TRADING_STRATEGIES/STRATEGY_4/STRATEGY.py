

class STRATEGY():
    def __init__(self, DATA_DF):
        self.DATA_DF    = DATA_DF
        self.DF         = self.STRATEGY()

    def STRATEGY(self):
        TAKE_PROFIT = 0.01  # Take Profit set to 1%
        STOP_LOSS   = 0.02  # Stop Loss set to 2%
        
        self.DATA_DF['SIGNAL'] = 'STATIC'
        self.DATA_DF['SIGNAL']          =   np.where((self.DATA_DF['EMA_18'].shift(1) <= self.DATA_DF['EMA_24'].shift(1)) & 
                                                    (self.DATA_DF['EMA_18'] > self.DATA_DF['EMA_24']), "LONG",  
                                            np.where((self.DATA_DF['EMA_18'].shift(1) >= self.DATA_DF['EMA_24'].shift(1)) & 
                                                    (self.DATA_DF['EMA_18'] < self.DATA_DF['EMA_24']), "SHORT",
                                                    "STATIC"))

        self.DATA_DF['TP']              = np.where((self.DATA_DF['SIGNAL'] == 'LONG'), (self.DATA_DF['OPEN'] * (1 + TAKE_PROFIT)), np.where((self.DATA_DF['SIGNAL'] == 'SHORT'), self.DATA_DF['OPEN'] * (1 - TAKE_PROFIT), 0))
        self.DATA_DF['SL']              = np.where((self.DATA_DF['SIGNAL'] == 'LONG'), (self.DATA_DF['OPEN'] * (1 - STOP_LOSS)), np.where((self.DATA_DF['SIGNAL'] == 'SHORT'), self.DATA_DF['OPEN'] * (1 + STOP_LOSS), 0))

        return self.DATA_DF
