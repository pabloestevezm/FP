import pandas as pd 
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy import signal
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


def stationary_or_not(df):
    """
    Apply the adfuller to compare if it's Stationary or not, to test the null hypothesis
    """
    stationaryCheck = lambda X: "Not-Stationary" if adfuller(X)[1] > 0.05 else "Stationary"
    return [(col,stationaryCheck(df[col])) for col in df.columns]


def modelling_AR(df, name):
    """
    Function to get the prediction model AR and apply to our DF
    """
    data_close = df[f'CLOSE_{name}']
    b, a = signal.butter(3, 1/10)
    filtrd_data_close = signal.filtfilt(b, a, data_close)
    df = pd.DataFrame({"X":data_close.to_numpy(),"Xf": filtrd_data_close},index=df.index)
    dr = df.index
    print(dr, dr.shape)
    realidad = df.loc[dr[:22808]]
    futuro = df.loc[dr[22808:]]
    predictions_AR = dict()

    for col in realidad.columns:
        train = realidad[col]
        test = futuro[col]

    # Entrena el modelo AR
    model_AR = AR(train)
    print(f"Entrenando con los datos desde la serie {col}")
    model_fit_AR = model_AR.fit(maxlag=10)
    
     # Predice los valores AR
    predictions_AR[col] = model_fit_AR.predict(start=len(train),
                                    end=len(train)+len(test)-1, dynamic=False)
      
    pred_AR = pd.DataFrame(predictions_AR)
    pred_AR['timing'] = futuro.index
    pred_AR.index = pd.DatetimeIndex(pred_AR.timing)
    pred_AR = pred_AR.drop(columns={'timing'})
    pred_AR

    for col in pred_AR.columns:
        mse = mean_squared_error(futuro[col], pred_AR[col])
        rmse = np.sqrt(mse)
        print(f"AR Model {col} ->  MSE={mse} RMSE={rmse}")

    
