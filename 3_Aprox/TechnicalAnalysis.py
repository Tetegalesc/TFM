import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        plt.plot(self.df['date'], self.df['close'], label='close')
        for period in self.timeperiods:
            self.df['SMA{}'.format(period)] = ta.SMA(self.df['close'], timeperiod=period)
            plt.plot(self.df['date'], self.df['SMA{}'.format(period)], label='SMA{}'.format(period))
        
        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('SMA')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/SMA.png')
        plt.close()

        

    # EMA. Exponential Moving Average
    def EMA(self):
        for period in self.timeperiods:
            self.df['EMA{}'.format(period)] = ta.EMA(self.df['close'], timeperiod=period)

        plt.plot(self.df['date'], self.df['close'], label='close')
        plt.plot(self.df['date'], self.df['SMA30'], label='SMA30')
        plt.plot(self.df['date'], self.df['EMA30'], label='EMA30')

        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('SMA vs EMA')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/SMAvsEMA.png')
        plt.close()

    # DEMA. Double Exponential Moving Average
    def DEMA(self):
        for period in self.timeperiods:
            self.df['DEMA{}'.format(period)] = ta.DEMA(self.df['close'], timeperiod=period)

        plt.plot(self.df['date'], self.df['close'], label='close')
        plt.plot(self.df['date'], self.df['EMA30'], label='EMA30')
        plt.plot(self.df['date'], self.df['DEMA30'], label='DEMA30')

        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('EMA vs DEMA')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/EMAvsDEMA.png')
        plt.close()

    # TEMA. Triple Exponential Moving Average
    def TEMA(self):
        for period in self.timeperiods:
            self.df['TEMA{}'.format(period)] = ta.TEMA(self.df['close'], timeperiod=period)

        plt.plot(self.df['date'], self.df['close'], label='close')
        plt.plot(self.df['date'], self.df['EMA30'], label='EMA30')
        plt.plot(self.df['date'], self.df['DEMA30'], label='DEMA30')
        plt.plot(self.df['date'], self.df['TEMA30'], label='TEMA30')

        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('EMA vs DEMA vs TEMA')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/EMAvsDEMAvsTEMA.png')
        plt.close()

    # KAMA. Kaufman Adaptive Moving Average
    def KAMA(self):
        for period in self.timeperiods:
            self.df['KAMA{}'.format(period)] = ta.KAMA(self.df['close'], timeperiod=period)

        plt.plot(self.df['date'], self.df['close'], label='close')
        plt.plot(self.df['date'], self.df['SMA30'], label='SMA30')
        plt.plot(self.df['date'], self.df['EMA30'], label='EMA30')
        plt.plot(self.df['date'], self.df['KAMA30'], label='KAMA30')

        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('SMA vs EMA vs KAMA')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/SMAvsEMAvsKAMA.png')
        plt.close()

    # MAMA. MESA Adaptive Moving Average
    def MAMA(self):
        self.df['MAMA'], self.df['FAMA'] = ta.MAMA(self.df['close'], fastlimit=0.5,slowlimit=0.05)

        plt.plot(self.df['date'], self.df['close'], label='close')
        plt.plot(self.df['date'], self.df['MAMA'], label='MAMA')
        plt.plot(self.df['date'], self.df['FAMA'], label='FAMA')

        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('MAMA')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/MAMA.png')
        plt.close()

    # MAVP. Moving Average with Variable Period
    def MAVP(self):
        self.df['MAVP'] = ta.MAVP(self.df['close'], self.df['date'], minperiod=2, maxperiod=30, matype=0)

        plt.plot(self.df['date'], self.df['close'], label='close')
        plt.plot(self.df['date'], self.df['MAVP'], label='MAVP')

        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('MAVP')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/MAVP.png')
        plt.close()

    # TRIMA. Triangular Moving Average
    def TRIMA(self):
        for period in self.timeperiods:
            self.df['TRIMA{}'.format(period)] = ta.TRIMA(self.df['close'], timeperiod=period)

        plt.plot(self.df['date'], self.df['close'], label='close')
        plt.plot(self.df['date'], self.df['SMA30'], label='SMA30')
        plt.plot(self.df['date'], self.df['EMA30'], label='EMA30')
        plt.plot(self.df['date'], self.df['KAMA30'], label='KAMA30')
        plt.plot(self.df['date'], self.df['TRIMA30'], label='TRIMA30')

        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('SMA vs EMA vs KAMA vs TRIMA')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/SMAvsEMAvsKAMAvsTRIMA.png')
        plt.close()

    # WMA. Weighted Moving Average
    def WMA(self):
        for period in self.timeperiods:
            self.df['WMA{}'.format(period)] = ta.WMA(self.df['close'], timeperiod=period)

        plt.plot(self.df['date'], self.df['close'], label='close')
        plt.plot(self.df['date'], self.df['SMA30'], label='SMA30')
        plt.plot(self.df['date'], self.df['EMA30'], label='EMA30')
        plt.plot(self.df['date'], self.df['KAMA30'], label='KAMA30')
        plt.plot(self.df['date'], self.df['TRIMA30'], label='TRIMA30')
        plt.plot(self.df['date'], self.df['WMA30'], label='WMA30')

        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('SMA vs EMA vs KAMA vs TRIMA vs WMA')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/SMAvsEMAvsKAMAvsTRIMAvsWMA.png')
        plt.close()

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

        plt.plot(self.df['date'], self.df['close'], label='close')
        plt.plot(self.df['date'], self.df['BBANDS_upper'], label='BBANDS_upper')
        plt.plot(self.df['date'], self.df['BBANDS_middle'], label='BBANDS_middle')
        plt.plot(self.df['date'], self.df['BBANDS_lower'], label='BBANDS_lower')

        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('Bollinger Bands')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/BBANDS.png')
        plt.close()

    # HT TRENDLINE. Hilbert Transform - Instantaneous Trendline
    def HT_TRENDLINE(self):
        self.df['HT_TRENDLINE'] = ta.HT_TRENDLINE(self.df['close'])

        plt.plot(self.df['date'], self.df['close'], label='close')
        plt.plot(self.df['date'], self.df['HT_TRENDLINE'], label='HT_TRENDLINE')

        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('HT Trendline')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/HT_TRENDLINE.png')
        plt.close()

    # MIDPOINT. MidPoint over Period
    def MIDPOINT(self):
        plt.plot(self.df['date'], self.df['close'], label='close')
        for period in self.timeperiods:
            self.df['MIDPOINT{}'.format(period)] = ta.MIDPOINT(self.df['close'], timeperiod=period)
            plt.plot(self.df['date'], self.df['MIDPOINT{}'.format(period)], label='MIDPOINT{}'.format(period))

        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('Midpoint over Period')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/MIDPOINT.png')
        plt.close()

    # MIDPRICE. MidPoint Price over period
    def MIDPRICE(self):
        plt.plot(self.df['date'], self.df['close'], label='close')
        for period in self.timeperiods:
            self.df['MIDPRICE{}'.format(period)] = ta.MIDPRICE(self.df['high'], self.df['low'], timeperiod=period)
            plt.plot(self.df['date'], self.df['MIDPRICE{}'.format(period)], label='MIDPRICE{}'.format(period))

        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('Midpoint Price over Period')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/MIDPRICE.png')
        plt.close()

    # SAR. Parabolic SAR
    def SAR(self):
        self.df['SAR'] = ta.SAR(self.df['high'], self.df['low'])

        plt.plot(self.df['date'], self.df['close'], label='close')
        plt.plot(self.df['date'], self.df['SAR'], label='SAR')

        plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

        plt.title('Parabolic SAR')
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.legend()

        plt.savefig('Results/0_Data/Results/SAR.png')
        plt.close()

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

        fig, axs = plt.subplots(1, 2, figsize=(15,6))

        axs[0].plot(self.df['date'], self.df['close'], label='close')
        axs[0].set(xlabel='Date', ylabel='Close')
        axs[0].set_title('Close')
        axs[0].axis([datetime(2020,1,1), datetime(2020,6,1), 50, 100])

        axs[1].plot(self.df['date'], self.df['MACD'], label='MACD')
        axs[1].plot(self.df['date'], self.df['MACD_Signal'], label='MACD Signal')
        axs[1].bar(self.df['date'], self.df['MACD_Hist'])

        axs[1].axis([datetime(2020,1,1), datetime(2020,6,1), -6, 6])
        axs[1].set_title('MACD')
        axs[1].set(xlabel='Date')
        axs[1].legend()

        plt.savefig('Results/0_Data/Results/MACD.png')
        plt.close()

        

    # ROC. Rate of Change
    def ROC(self):
        fig, axs = plt.subplots(1, 2, figsize=(15,6))

        axs[0].plot(self.df['date'], self.df['close'], label='close')
        axs[0].set(xlabel='Date', ylabel='Close')
        axs[0].set_title('Close')
        axs[0].axis([datetime(2020,1,1), datetime(2020,6,1), 50, 100])

        for period in self.timeperiods:
            self.df['ROC{}'.format(period)] = ta.ROC(self.df['close'], timeperiod=period)
            axs[1].plot(self.df['date'], self.df['ROC{}'.format(period)], label='ROC{}'.format(period))

        axs[1].axis([datetime(2020,1,1), datetime(2020,6,1), -50, 50])

        axs[1].set(xlabel='Date')
        axs[1].set_title('ROC')
        axs[1].legend()

        plt.savefig('Results/0_Data/Results/ROC.png')
        plt.close()

    # STOCH. Stochastic
    def STOCH(self):
        self.df['STOCH_Fast'], self.df['STOCH_Slow'] = ta.STOCH(self.df['high'], self.df['low'], self.df['close'])

        fig, axs = plt.subplots(1, 2, figsize=(15,6))

        axs[0].plot(self.df['date'], self.df['close'], label='close')
        axs[0].set(xlabel='Date', ylabel='Close')
        axs[0].set_title('Close')
        axs[0].axis([datetime(2020,1,1), datetime(2020,6,1), 50, 100])


        axs[1].plot(self.df['date'], self.df['STOCH_Fast'], label='STOCH_Fast')
        axs[1].plot(self.df['date'], self.df['STOCH_Slow'], label='STOCH_Slow')

        axs[1].axis([datetime(2020,1,1), datetime(2020,6,1), 0, 100])

        axs[1].set(xlabel='Date')
        axs[1].set_title('STOCH')
        axs[1].legend()

        plt.savefig('Results/0_Data/Results/STOCH.png')
        plt.close()

    # RSI. Relative Strength Index
    def RSI(self):
        fig, axs = plt.subplots(1, 2, figsize=(15,6))

        axs[0].plot(self.df['date'], self.df['close'], label='close')
        axs[0].set(xlabel='Date', ylabel='Close')
        axs[0].set_title('Close')
        axs[0].axis([datetime(2020,1,1), datetime(2020,6,1), 50, 100])

        for period in self.timeperiods:
            self.df['RSI{}'.format(period)] = ta.RSI(self.df['close'], timeperiod=period)
            axs[1].plot(self.df['date'], self.df['RSI{}'.format(period)], label='RSI{}'.format(period))

        axs[1].axis([datetime(2020,1,1), datetime(2020,6,1), 0, 100])

        axs[1].set(xlabel='Date')
        axs[1].set_title('RSI')
        axs[1].legend()

        plt.savefig('Results/0_Data/Results/RSI.png')
        plt.close()

    # ADX. Average Directional Movement Index
    def ADX(self):
        fig, axs = plt.subplots(1, 2, figsize=(15,6))

        axs[0].plot(self.df['date'], self.df['close'], label='close')
        axs[0].set(xlabel='Date', ylabel='Close')
        axs[0].set_title('Close')
        axs[0].axis([datetime(2020,1,1), datetime(2020,6,1), 50, 100])

        for period in self.timeperiods:
            self.df['ADX{}'.format(period)] = ta.ADX(self.df['high'], self.df['low'], self.df['close'], timeperiod=period)
            axs[1].plot(self.df['date'], self.df['ADX{}'.format(period)], label='ADX{}'.format(period))

        axs[1].axis([datetime(2020,1,1), datetime(2020,6,1), 0, 100])

        axs[1].set(xlabel='Date')
        axs[1].set_title('ADX')
        axs[1].legend()

        plt.savefig('Results/0_Data/Results/ADX.png')
        plt.close()