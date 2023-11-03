import pandas as pd
import numpy as np
from datetime import datetime

# Web Scrapping
import bs4 as bs
import requests

# Data Preprocessing
from yahoo_fin import stock_info as si
# Technical Analysis
import talib as ta
from TechnicalAnalysis import TA

#######################################################################################################################
# Data Loader
#######################################################################################################################

def DataLoader():
    SP500_tickers = get_SP500_tickers()
    # Get Data form all SP500_tickers

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

    tickers_df = []

    for ticker in SP500_tickers:
        print('Loading {} Data'.format(ticker))
        # Loading Data
        df = LoadData(ticker)

        # Data Preprocessing
        DataPreprocessing(df, timeperiods, features=features_to_scale, hor=5)

        # Appending DataFrame
        tickers_df.append(df)

    # Getting all df together in one single df
    all_tickers_df = pd.concat(tickers_df)

    # Sorting Data by data
    all_tickers_df.sort_values(by='date', inplace=True)
    # Reindexing dataframe
    all_tickers_df.reset_index(drop=True, inplace=True)

    # Filtering Data only when the ticker was in the SP500
    final_df = OnlySP500(all_tickers_df)

    # Deleting Columns
    DelColumns(final_df)

    return final_df


#######################################################################################################################
# Get SP500 tickers
#######################################################################################################################

def get_SP500_tickers():
    # Accesing wikipedia table
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class' : 'wikitable sortable'})

    # Getting every ticker
    SP500_tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        SP500_tickers.append(ticker)

    # Removing \n
    SP500_tickers = [t.replace('\n', '') for t in SP500_tickers]
    
    # Replacing '.' for '-'
    SP500_tickers = [t.replace('.', '-') for t in SP500_tickers]

    return SP500_tickers

#######################################################################################################################
# Obtain data
#######################################################################################################################

def LoadData(Ticker, start_date = None, end_date = None):
    df = si.get_data(ticker = Ticker, start_date = start_date, end_date = end_date, interval = '1d')
    return df

#######################################################################################################################
#######################################################################################################################
# Data Preprocessing
#######################################################################################################################
#######################################################################################################################

def DataPreprocessing(df, timeperiods, features, hor=5):
    DateIndex(df)
    FeatureEngineering(df, timeperiods, hor)
    DataScaling(df, features)
    NaNValues(df)

#######################################################################################################################
# Date Index
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

def DataScaling(df, features):
    ## Scaling volume
    df['SMA30_vol'] = ta.SMA(df['volume'], timeperiod=30)
    df['volume'] = (df['volume'] - df['SMA30_vol']) / df['SMA30_vol']
    
    ## Scaling features
    df['SMA30_day_before'] = df['SMA30'].shift(1)
    for feat in features:
        df[feat] = (df[feat] - df['SMA30_day_before']) / df['SMA30_day_before']

    # Deleting columns
    df.drop(['SMA30_vol', 'SMA30_day_before'], axis=1, inplace=True)

#######################################################################################################################
# NaN Values
#######################################################################################################################

def NaNValues(df):
    df.dropna(inplace=True)

#######################################################################################################################
# Only Data in SP500
#######################################################################################################################

# Filtering Data in SP500 depending on date
def OnlySP500(df):
    # Getting the tickers that were in the SP500 by date
    tickers_in_SP500 = pd.read_csv('S&P 500 Historical Components & Changes(08-01-2023).csv')
    tickers_in_SP500['date'] = pd.to_datetime(tickers_in_SP500['date'])
    tickers_in_SP500['tickers'] = tickers_in_SP500['tickers'].apply(lambda x: sorted(x.split(','))).to_list()
    tickers_in_SP500['tickers'] = [[t.replace('.', '-') for t in tickers_list] for tickers_list in tickers_in_SP500['tickers']]

    # Merging the df with the one calculated below
    Only_SP500_df = tickers_in_SP500.merge(df, on='date')

    # Now keep only the ones that the company is in tickers_in_SP500
    Only_SP500_df = Only_SP500_df[Only_SP500_df.apply(lambda row: row['ticker'] in row['tickers'], axis=1)]

    # Sorting Data by date and reseting index
    Only_SP500_df.sort_values(by='date', inplace=True)
    Only_SP500_df.reset_index(drop=True, inplace=True)

    return Only_SP500_df


#######################################################################################################################
# Deleting columns
#######################################################################################################################

# Delete date and ticker columns
def DelColumns(df):
    df.drop(['ticker', 'tickers'], axis=1, inplace=True)