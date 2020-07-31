# FinalProject idea




-Dataset

https://www.kaggle.com/amin233/forex-top-currency-pairs-20002020


-Specialized in one pair (EUR/USD) 

## Topics

-Tratamiento de los datasets

-Modelos de series temporales (AR y ARIMA)

-NN modelo Long Short-Term Memory

-Métricas de las predicciones de cada par respecto a cada modelo y visualización de los datos obtenidos

Aplico los modelos respecto al precio de cierre para hacer las predicciones.
Estas predicciones van a ser sobre el mes de Diciembre de 2019

- Previo al análisis de los modelos autorregresivos, obtengo:
Prueba de Dickey-Fuller (no estacionaria)
signal-butter : filtro lowpass


## Modelo AR


- De los valores obtenidos del signal-butter, junto con los valores del GT
realizo las primeras predicciones con este modelo
Realiza unas predicciones lineales X(GT) y Xf(s.butter)


## Modelo ARIMA


- Mismo planteamiento que con el modelo AR, sólo que el order(p,d,q) que
hay que pasar al modelo me ha limitado mucho a la hora de poder cambiar las
predicciones