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
import talib as ta
from TechnicalAnalysis import TA

def DataLoader():
    # Creating Folder
    Data_results_path = 'Results/0_Data/Results'
    os.makedirs(Data_results_path, exist_ok=True)

    # Loading Data
    df = LoadData(Ticker='AAPL')
    mpf.plot(df,type='line',volume=True, savefig='Results/0_Data/Results/Price_plot.png')

    # Data Preprocessing
    timeperiods = np.arange(5,35,5)
    features_to_scale = ['open', 'high', 'low', 'close', 'adjclose', 'avg_price',
                         'SMA5', 'SMA10', 'SMA15', 'SMA20', 'SMA25', 'SMA30',
                         'EMA5', 'EMA10', 'EMA15', 'EMA20', 'EMA25', 'EMA30',
                         'DEMA5', 'DEMA10', 'DEMA15', 'DEMA20', 'DEMA25', 'DEMA30',
                         'TEMA5', 'TEMA10', 'TEMA15', 'TEMA20', 'TEMA25', 'TEMA30',
                         'KAMA5', 'KAMA10', 'KAMA15', 'KAMA20', 'KAMA25', 'KAMA30',
                         'MAMA', 'FAMA', 'MAVP',
                         'TRIMA5', 'TRIMA10', 'TRIMA15', 'TRIMA20', 'TRIMA25', 'TRIMA30',
                         'WMA5', 'WMA10', 'WMA15', 'WMA20', 'WMA25', 'WMA30',
                         'BBANDS_upper', 'BBANDS_middle', 'BBANDS_lower', 'HT_TRENDLINE',
                         'MIDPOINT5', 'MIDPOINT10', 'MIDPOINT15', 'MIDPOINT20', 'MIDPOINT25', 'MIDPOINT30',
                         'MIDPRICE5', 'MIDPRICE10', 'MIDPRICE15', 'MIDPRICE20', 'MIDPRICE25', 'MIDPRICE30',
                         'SAR']
    DataPreprocessing(df, timeperiods, features_to_scale, hor=5)

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

def DataPreprocessing(df, timeperiods, features_to_scale, hor=5):
    DateIndex(df)
    FeatureEngineering(df, timeperiods, hor)
    DataScaling(df, features_to_scale)
    NaNValues(df)
    ResetIndex(df)

#######################################################################################################################
# Data Index
#######################################################################################################################

def DateIndex(df):
    df.reset_index(inplace=True)
    df.rename({'index' : 'date'}, axis=1, inplace=True)

#######################################################################################################################
# Feature Engineering
#######################################################################################################################

def FeatureEngineering(df, timeperiods, hor=5):
    Volume(df)
    CreateGain(df, hor=hor)
    TA(df=df, timeperiods=timeperiods).ALL()

# Volume

def Volume(df):
    # Average Price of the day
    df['avg_price'] = df[['open', 'high', 'low', 'close']].mean(axis=1)
    # Multiply avg_price and volume
    df['volume'] = df['volume'] * df['avg_price']

# Gain

def CreateGain(df, hor=5):
    # Close price after "hor" days
    df['close_{}_days_after'.format(hor)] = df['close'].shift(-hor)
    # Gain after "hor" days
    df['gain'] = (df['close_{}_days_after'.format(hor)] - df['close']) / df['close']
    # Delete "close_{}_days_after" column
    df.drop('close_{}_days_after'.format(hor), axis=1, inplace=True)

#######################################################################################################################
# Data Scaling
#######################################################################################################################

def DataScaling(df,features):
    # Volume
    df['SMA30_vol'] = ta.SMA(df['volume'], timeperiod=30)
    df['volume'] = (df['volume'] - df['SMA30_vol']) / df['SMA30_vol']

    # Features
    df['SMA30_day_before'] = df['SMA30'].shift(1)
    for feat in features:
        df[feat] = (df[feat] - df['SMA30_day_before']) / df['SMA30_day_before']

    # Delete columns
    df.drop(['SMA30_vol', 'SMA30_day_before'], axis=1, inplace=True)

    # Plot Scale Data
    plt.plot(df['close'])

    plt.title('Scaled Data')
    plt.xlabel('Date')
    plt.ylabel('Close price')

    plt.savefig('Results/0_Data/Results/Scaled_Data.png')


#######################################################################################################################
# NaN Values
#######################################################################################################################

def NaNValues(df):
    df.dropna(inplace=True)

#######################################################################################################################
# Reset Index
#######################################################################################################################

def ResetIndex(df):
    df.sort_values(by='date', inplace=True)
    df.reset_index(drop=True, inplace=True)

#######################################################################################################################
# Delete Columns
#######################################################################################################################
def DelColumns(df):
    df.drop(['ticker'], axis=1, inplace=True)