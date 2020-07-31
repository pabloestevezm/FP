import pandas as pd 
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy import signal
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, LSTM, Dense, Flatten, Conv2D, MaxPooling2D


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

        # Entrena el modelo ARIMA
        model_ARIMA = ARIMA(train, order=(0,0,1))
        print(f"Entrenando con los datos desde la serie {col}")
        model_fit_ARIMA = model_ARIMA.fit(maxlag=4)
        
        # Predice los valores ARIMA
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




def create_dataset(df, name, look_back=10):
    """
    This function is for giving my df the 3d with the timesteps to calculate the NN
    """
    df_copy = df.copy()
    df_copy = df_copy.drop(f'close_{name}', axis=1)
    dataX, dataY = [], []
    for i in range(len(df)-look_back-1):
        a = df[i:(i+look_back)]
        dataX.append(a)
    return np.array(dataX), np.array(dataY)



def modelling_LSTM(df, name, look_back=4):
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaler = x_scaler.fit(df.values.astype('float32'))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = y_scaler.fit(df[f'close_{name}'].values.astype('float32').reshape(-1,1))
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    X, y = create_dataset(scaled, name, look_back)
    #print(X.shape, X)
    y = scaled[look_back:-1]
    y = scaled[f'close_{name}']
    SPLIT = 0.9
    train_size = int(len(X) * SPLIT)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    model = Sequential()
    model.add(LSTM(20, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(10, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model_LSTM = model.fit(X_train, y_train, epochs=4, batch_size=1)
    return model_LSTM, model, X_test, y_test, y_scaler







def get_metrics(df, df_orig, name):

    data_close = df_orig[f'CLOSE_{name}']
    b, a = signal.butter(3, 1/10)
    filtrd_data_close = signal.filtfilt(b, a, data_close)
    df2 = pd.DataFrame({"X":data_close.to_numpy(),"Xf": filtrd_data_close},index=df_orig.index)
    dr = df2.index
    realidad = df2.loc[dr[:22808]]
    futuro = df2.loc[dr[22808:]]
    for col in futuro.columns:
        mse = mean_squared_error(futuro[col], df[col])
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((futuro[col] - df[col]) / futuro[col])) * 100
        mae = mean_absolute_error(futuro[col], df[col])
        print(f"AR Model {col} ->  MSE={mse}  RMSE={rmse}  MAPE={mape}  MAE={mae}")
    
    return mse, rmse, mape, mae
    


def get_pred_LSTM(model_LSTM, model, X_test, y_test, y_scaler):
    pred = model.predict(X_test)
    prediction_np = np.reshape(pred, (2478, 4))
    y_test2 = y_test.values.reshape(2483,1)
    prediction = pd.DataFrame(prediction_np)
    predict = []
    for i in range(len(prediction)):
        x = (prediction.iloc[i].sum())/(len(prediction.iloc[i]))
        predict.append(x)
    
    prediction['pred']= predict
    prediction_clean = prediction.drop(columns={0,1,2,3})
    pred = y_scaler.inverse_transform(prediction_clean)
    close = y_scaler.inverse_transform(y_test2)

    predictions = pd.DataFrame()
    predictions['predicted'] = pd.Series(np.reshape(pred, (pred.shape[0])))
    predictions['GT'] = pd.Series(np.reshape(close, (close.shape[0])))
    
    return predictions
