import pandas as pd
import numpy as np
import datetime

def red_and_ind(df):
    """
    This function returns a DataFrame with time index, and reduce the DataFrame to only take 2019 values
    """
    df['all_dates'] = pd.to_datetime(df['DATE_TIME'])
    df['dates'] = (df['all_dates'] > '2018-12-31 23:45:00') & (df['all_dates'] <= '2020-01-01 00:00:00')
    df = df[df['dates'] == True]
    df.index = pd.DatetimeIndex(df.DATE_TIME)
    df = df.drop(columns={'DATE_TIME', 'all_dates', 'dates'})
    return df


def renaming(df, name):
    """
    This function will be useful when we concat differents DFs to help us to difference between pairs
    """
    df = df.rename(columns={'HIGH':f'HIGH_{name}', 'LOW':f'LOW_{name}', 'OPEN':f'OPEN_{name}', 'CLOSE':f'CLOSE_{name}'})
    return df


#predict_GRAD500 = predict_GRAD500.rename(columns={"index": "id", 0: "price"})