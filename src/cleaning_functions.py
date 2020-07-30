import pandas as pd
import numpy as np
import datetime
from statsmodels.tsa.stattools import adfuller

def red_and_ind(df):
    """
    This function returns a DataFrame with time index, and reduce the DataFrame to only take 2019 values
    """
    df['all_dates'] = pd.to_datetime(df['DATE_TIME'])
    df['dates'] = (df['all_dates'] > '2018-12-31 23:45:00') & (df['all_dates'] <= '2020-01-01 00:00:00')
    df = df[df['dates'] == True]
    df.index = pd.DatetimeIndex(df.DATE_TIME)
    df = df.drop(columns={'DATE_TIME', 'all_dates', 'dates'})
    df['DIFF']= df['CLOSE']-df['OPEN']
    return df


def get_sample(df, timestart, timeend):
    """
    This function get dates to return DF filtered from the clean csv
    """
    df['all_dates'] = pd.to_datetime(df['DATE_TIME'])
    df['dates'] = (df['all_dates'] > f'{timestart}') & (df['all_dates'] <= f'{timeend}')
    df = df[df['dates'] == True]
    df.index = pd.DatetimeIndex(df.DATE_TIME)
    df = df.drop(columns={'DATE_TIME', 'all_dates', 'dates'})
    return df



def red_dates(df, name, timestart, timeend):
    df['dates'] = (df['DATE_TIME'] > f'{timestart}') & (df['DATE_TIME'] <= f'{timeend}')
    df = df[df['dates'] == True]
    df = df.drop(columns={'dates'})
    return df



def red_LSTM(df, name, timestart, timeend):
    original_cols = ["DATE_TIME", "HIGH", "LOW", "OPEN", "CLOSE"]
    cols_name = ["time", f"high_{name}", f"low_{name}", f"open_{name}", f"close_{name}"]
    df['dates'] = (df['DATE_TIME'] > f'{timestart}') & (df['DATE_TIME'] <= f'{timeend}')
    df = df[df['dates'] == True]
    df = df.drop(columns={'dates'})
    df.rename(columns=dict(zip(original_cols, cols_name)), inplace=True)
    df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
    df.set_index("time", inplace=True)
    df = df.reindex(columns=cols_name[1:])
    return df




def renaming(df, name):
    """
    This function will be useful when we concat differents DFs to help us to difference between pairs
    """
    df = df.rename(columns={'HIGH':f'HIGH_{name}', 'LOW':f'LOW_{name}', 'OPEN':f'OPEN_{name}', 'CLOSE':f'CLOSE_{name}', 'DIFF':f'DIFF_{name}'})
    return df


def renam_LSTM(df, name):
    df = df.rename(columns={'HIGH':f'HIGH_{name}', 'LOW':f'LOW_{name}', 'OPEN':f'OPEN_{name}', 'CLOSE':f'CLOSE_{name}'})
    return df
