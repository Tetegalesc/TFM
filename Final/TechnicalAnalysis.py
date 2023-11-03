import os
import pandas as pd
import numpy as np

from datetime import datetime
import time
from yahoo_fin import stock_info as si

import talib as ta

class TA():

    def __init__(self, df, timeperiods):
        
        self.df = df
        self.timeperiods = timeperiods

    #===================================================================================================================
    #===================================================================================================================
    #===================================================================================================================
    # ALL
    #===================================================================================================================
    #===================================================================================================================
    #===================================================================================================================
    
    def ALL(self):
        self.OverlapStudies()
        self.MomentumIndicators()

    #===================================================================================================================
    #===================================================================================================================
    # Overlap Studies
    #===================================================================================================================
    #===================================================================================================================

    def OverlapStudies(self):
        self.MA()
        self.TRENDS()

    #===================================================================================================================
    # Moving Averages
    #===================================================================================================================

    def MA(self):
        self.SMA()
        self.EMA()
        self.DEMA()
        self.TEMA()
        self.KAMA()
        self.MAMA()
        self.MAVP()
        self.TRIMA()
        self.WMA()

    # SMA. Simple Moving Average
    def SMA(self):
        
        for period in self.timeperiods:
            self.df['SMA{}'.format(period)] = ta.SMA(self.df['close'], timeperiod=period)        

    # EMA. Exponential Moving Average
    def EMA(self):
        for period in self.timeperiods:
            self.df['EMA{}'.format(period)] = ta.EMA(self.df['close'], timeperiod=period)

    # DEMA. Double Exponential Moving Average
    def DEMA(self):
        for period in self.timeperiods:
            self.df['DEMA{}'.format(period)] = ta.DEMA(self.df['close'], timeperiod=period)

    # TEMA. Triple Exponential Moving Average
    def TEMA(self):
        for period in self.timeperiods:
            self.df['TEMA{}'.format(period)] = ta.TEMA(self.df['close'], timeperiod=period)

    # KAMA. Kaufman Adaptive Moving Average
    def KAMA(self):
        for period in self.timeperiods:
            self.df['KAMA{}'.format(period)] = ta.KAMA(self.df['close'], timeperiod=period)

    # MAMA. MESA Adaptive Moving Average
    def MAMA(self):
        self.df['MAMA'], self.df['FAMA'] = ta.MAMA(self.df['close'], fastlimit=0.5,slowlimit=0.05)

    # MAVP. Moving Average with Variable Period
    def MAVP(self):
        self.df['MAVP'] = ta.MAVP(self.df['close'], self.df['date'], minperiod=2, maxperiod=30, matype=0)

    # TRIMA. Triangular Moving Average
    def TRIMA(self):
        for period in self.timeperiods:
            self.df['TRIMA{}'.format(period)] = ta.TRIMA(self.df['close'], timeperiod=period)

    # WMA. Weighted Moving Average
    def WMA(self):
        for period in self.timeperiods:
            self.df['WMA{}'.format(period)] = ta.WMA(self.df['close'], timeperiod=period)

    #===================================================================================================================
    # Trends
    #===================================================================================================================

    def TRENDS(self):
        self.BBANDS()
        self.HT_TRENDLINE()
        self.MIDPOINT()
        self.MIDPRICE()
        self.SAR()
        self.ADX()

    # BBANDS. Bollinger Bands
    def BBANDS(self):
        self.df['BBANDS_upper'], self.df['BBANDS_middle'], self.df['BBANDS_lower'] = ta.BBANDS(self.df['close'], timeperiod=5)

    # HT TRENDLINE. Hilbert Transform - Instantaneous Trendline
    def HT_TRENDLINE(self):
        self.df['HT_TRENDLINE'] = ta.HT_TRENDLINE(self.df['close'])

    # MIDPOINT. MidPoint over Period
    def MIDPOINT(self):
        
        for period in self.timeperiods:
            self.df['MIDPOINT{}'.format(period)] = ta.MIDPOINT(self.df['close'], timeperiod=period)

    # MIDPRICE. MidPoint Price over period
    def MIDPRICE(self):
        
        for period in self.timeperiods:
            self.df['MIDPRICE{}'.format(period)] = ta.MIDPRICE(self.df['high'], self.df['low'], timeperiod=period)

    # SAR. Parabolic SAR
    def SAR(self):
        self.df['SAR'] = ta.SAR(self.df['high'], self.df['low'])

    #===================================================================================================================
    #===================================================================================================================
    # Momentum Indicators
    #===================================================================================================================
    #===================================================================================================================

    def MomentumIndicators(self):
        self.MACD()
        self.ROC()
        self.STOCH()
        self.RSI()

    # MACD. Moving Average Convergence Divergence
    def MACD(self):
        self.df['MACD'], self.df['MACD_Signal'], self.df['MACD_Hist'] = ta.MACD(self.df['close'], fastperiod=12, slowperiod=26, signalperiod=9)        

    # ROC. Rate of Change
    def ROC(self):

        for period in self.timeperiods:
            self.df['ROC{}'.format(period)] = ta.ROC(self.df['close'], timeperiod=period)

    # STOCH. Stochastic
    def STOCH(self):
        self.df['STOCH_Fast'], self.df['STOCH_Slow'] = ta.STOCH(self.df['high'], self.df['low'], self.df['close'])

    # RSI. Relative Strength Index
    def RSI(self):

        for period in self.timeperiods:
            self.df['RSI{}'.format(period)] = ta.RSI(self.df['close'], timeperiod=period)

    # ADX. Average Directional Movement Index
    def ADX(self):

        for period in self.timeperiods:
            self.df['ADX{}'.format(period)] = ta.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)