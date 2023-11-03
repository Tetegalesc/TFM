import os
import pandas as pd
import numpy as np
from datetime import datetime

# Plot Data
import mplfinance as mpf
import matplotlib.pyplot as plt

# Data Loading
from yahoo_fin import stock_info as si
# Technical Analysis
from TechnicalAnalysis import TA

def DataLoader():
    # Creating Folder
    Data_results_path = 'Results/0_Data/Results'
    os.makedirs(Data_results_path, exist_ok=True)

    # Loading Data
    df = LoadData(Ticker='AAPL')
    mpf.plot(df,type='line',volume=True, savefig='Results/0_Data/Results/Price_plot.png')

    # Data Preprocessing
    timeperiods = np.arange(5,20,5)
    DataPreprocessing(df, timeperiods, hor=5)

    # Plot Data
    PlotData(df,timeperiods)

    # Delete columns
    DelColumns(df)

    return df

#######################################################################################################################
# Obtain data
#######################################################################################################################

def LoadData(Ticker, start_date = None, end_date = datetime(2023,8,1)):
    df = si.get_data(ticker = Ticker, start_date = start_date, end_date = end_date, interval = '1d')
    return df

#######################################################################################################################
#######################################################################################################################
# Data Preprocessing
#######################################################################################################################
#######################################################################################################################

def DataPreprocessing(df, timeperiods, hor=5):
    FeatureEngineering(df, timeperiods, hor)
    NaNValues(df)

#######################################################################################################################
# Feature Engineering
#######################################################################################################################

def FeatureEngineering(df, timeperiods, hor=5):
    CreateGain(df, hor=hor)
    TA(df=df, timeperiods=timeperiods).ALL()

# Gain

def CreateGain(df, hor=5):
    # Close price after "hor" days
    df['close_{}_days_after'.format(hor)] = df['close'].shift(-hor)
    # Gain after "hor" days
    df['gain'] = (df['close_{}_days_after'.format(hor)] - df['close']) / df['close']
    # Delete "close_{}_days_after" column
    df.drop('close_{}_days_after'.format(hor), axis=1, inplace=True)

#######################################################################################################################
# NaN Values
#######################################################################################################################

def NaNValues(df):
    df.dropna(inplace=True)

#######################################################################################################################
# Delete Columns
#######################################################################################################################
def DelColumns(df):
    df.drop(['ticker'], axis=1, inplace=True)

#######################################################################################################################
#######################################################################################################################
# Plot Data
#######################################################################################################################
#######################################################################################################################

def PlotData(df,timeperiods):
    plotOverlap(df,timeperiods)
    plotMomentum(df)

#######################################################################################################################
# Plot OverlapStudies
#######################################################################################################################
def plotOverlap(df,timeperiods):
    plotMA(df,timeperiods)

#======================================================================================================================
# Plot MA
#======================================================================================================================
def plotMA(df,timeperiods):
    plotSMA(df,timeperiods)
    plotEMA(df)

def plotSMA(df,timeperiods):
    plt.plot(df['close'], 'k', label = 'close')

    for period in timeperiods:
        plt.plot(df['SMA{}'.format(period)], label='SMA{}'.format(period))

    plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

    plt.title('SMA')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.legend()

    plt.savefig('Results/0_Data/Results/SMA.png')
    plt.close()

def plotEMA(df):
    plt.plot(df['close'], 'k', label = 'close')

    plt.plot(df['SMA10'], 'r', label = 'SMA10')
    plt.plot(df['EMA10'], 'r--', label = 'EMA10')

    plt.axis([datetime(2020,1,1), datetime(2020,6,1), 50, 85])

    plt.title('SMA vs EMA')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.legend()

    plt.savefig('Results/0_Data/Results/SMAvsEMA.png')
    plt.close()

#######################################################################################################################
# Plot Momentum Indicators
#######################################################################################################################

def plotMomentum(df):
    plotRSI(df)
    plotMACD(df)

def plotRSI(df):
    plt.plot(df['RSI'])

    plt.axis([datetime(2020,1,1), datetime(2020,6,1), 10, 100])

    plt.title('RSI')
    plt.xlabel('Date')
    plt.ylabel('RSI')

    plt.savefig('Results/0_Data/Results/RSI.png')
    plt.close()

def plotMACD(df):
    plt.plot(df['MACD'], 'k', label='MACD')
    plt.plot(df['MACD_Signal'], 'r', label='MACD Signal')
    plt.bar(df.index, df['MACD_Hist'])

    plt.axis([datetime(2020,1,1), datetime(2020,6,1), -6, 6])

    plt.title('MACD')
    plt.xlabel('Date')

    plt.legend()

    plt.savefig('Results/0_Data/Results/MACD.png')
    plt.close()