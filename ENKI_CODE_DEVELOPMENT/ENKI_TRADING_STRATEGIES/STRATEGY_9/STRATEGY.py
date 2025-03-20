

############  AI  ############


#FILE	                                    STRATEGY	                        TP_SL	                END_BALANCE__1	END_BALANCE__2	END_BALANCE__3
#BTCUSDT -- 1D -- amazon/chronos-t5-small	1  --  4  --  STOCH_10	            TP_AI_1  --  SL_ATR_1	111,3201.9	    1.25	            1,361,874.11




class CONDITIONS():
    def __init__(self):
        self.DF         = self.CONDITIONS()

    def CONDITIONS(self):
        return ['ADAUSDT', '1D', 'LARGE', [0.75,  3,  'AVG_VOL_P_GAIN_3_2'],   ['TP_AI_2',  'SL_ATR_2'], 1]
