import pandas as pd 
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy import signal
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


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
    df2 = pd.DataFrame({"X":data_close.to_numpy(),"Xf": filtrd_data_close},index=df.index)
    dr = df2.index
    realidad = df2.loc[dr[:22808]]
    futuro = df2.loc[dr[22808:]]
    predictions_AR = dict()

    for col in realidad.columns:
        train = realidad[col]
        test = futuro[col]

        # Entrena el modelo AR
        model_AR = AR(train)
        print(f"Entrenando con los datos desde la serie {col}")
        model_fit_AR = model_AR.fit(maxlag=4)
        
        # Predice los valores AR
        predictions_AR[f'{col}_prediction'] = model_fit_AR.predict(start=len(train),
                                        end=len(train)+len(test)-1, dynamic=False)
      
    pred_AR = pd.DataFrame(predictions_AR)
    pred_AR.index = futuro.index

    AR_predictions = pd.DataFrame({
    "GT":futuro.X,
    "X":pred_AR.X_prediction,
    "Xf":pred_AR.Xf_prediction,
    "diff_X": futuro.X - pred_AR.X_prediction,
    "diff_Xf":futuro.X - pred_AR.Xf_prediction},index=futuro.index)

    return AR_predictions



def modelling_ARIMA(df, name):
    """
    Function to get the prediction model AR and apply to our DF
    """
    data_close = df[f'CLOSE_{name}']
    b, a = signal.butter(3, 1/10)
    filtrd_data_close = signal.filtfilt(b, a, data_close)
    df2 = pd.DataFrame({"X":data_close.to_numpy(),"Xf": filtrd_data_close},index=df.index)
    dr = df2.index
    realidad = df2.loc[dr[:22808]]
    futuro = df2.loc[dr[22808:]]
    predictions_ARIMA = dict()

    for col in realidad.columns:
        train = realidad[col]
        test = futuro[col]

        # Entrena el modelo AR
        model_ARIMA = ARIMA(train, order=(0,0,1))
        print(f"Entrenando con los datos desde la serie {col}")
        model_fit_ARIMA = model_ARIMA.fit(maxlag=4)
        
        # Predice los valores AR
        predictions_ARIMA[f'{col}_prediction'] = model_fit_ARIMA.predict(start=len(train),
                                        end=len(train)+len(test)-1, dynamic=False)
      
    pred_ARIMA = pd.DataFrame(predictions_ARIMA)
    pred_ARIMA.index = futuro.index

    ARIMA_predictions = pd.DataFrame({
    "GT":futuro.X,
    "X":pred_ARIMA.X_prediction,
    "Xf":pred_ARIMA.Xf_prediction,
    "diff_X": futuro.X - pred_ARIMA.X_prediction,
    "diff_Xf":futuro.X - pred_ARIMA.Xf_prediction},index=futuro.index)

    return ARIMA_predictions







def get_metrics(df, df_orig, name):

    data_close = df_orig[f'CLOSE_{name}']
    b, a = signal.butter(3, 1/10)
    filtrd_data_close = signal.filtfilt(b, a, data_close)
    df2 = pd.DataFrame({"X":data_close.to_numpy(),"Xf": filtrd_data_close},index=df_orig.index)
    dr = df2.index
    realidad = df2.loc[dr[:22808]]
    futuro = df2.loc[dr[22808:]]
    print(futuro.shape)
    for col in futuro.columns:
        mse = mean_squared_error(futuro[col], df[col])
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((futuro[col] - df[col]) / futuro[col])) * 100
        mae = mean_absolute_error(futuro[col], df[col])
        print(f"AR Model {col} ->  MSE={mse}  RMSE={rmse}  MAPE={mape}  MAE={mae}")
    
    return mse, rmse, mape, mae
    

    
