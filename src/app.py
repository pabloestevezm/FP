
from flask import Flask

app = Flask(__name__)


@app.route("/currencies_pairs")
def currencies_opts():
    currs={
        "opt1":"EUR-USD",
        "opt2":"EUR-JPY",
        "opt3":"EUR-CHF",
        "opt4":"USD-JPY",
        "opt5":"USD-CAD",
        "opt6":"USD-CHF",
        "opt7":"AUD-USD",
    }
    return currs


@app.route("/currencies_pairs/EUR-USD")
def metrics_eur_usd():
    metrics={"AR model": {
        "MSE":3.614101692169201e-05,
        "RMSE":0.006007549111511034,
        "MAE":0.00502701275,
        "MAPE":0.45287004293971334
    },
    "ARIMA":{
        "MSE":0.00010233317821826405,
        "RMSE":0.010115986270169807,
        "MAE":0.009184528267,
        "MAPE":0.8281034865195951
    },
    "LSTM":{
        "MSE":6.007233489449766e-08,
        "MAE":0.00019088951835750213
    }}
   
    return metrics


@app.route("/currencies_pairs/USD-JPY")
def metrics_usd_jpy():
    metrics={"AR model": {
        "MSE":0.22565868861232463,
        "RMSE":0.47503546037356476,
        "MAE":0.3580792364756238,
        "MAPE":0.3898679934827795
    },
    "ARIMA":{
        "MSE":"no values",
        "RMSE":"no values",
        "MAE":"no values",
        "MAPE":"no values"
    },
    "LSTM":{
        "MSE":0.0023590851498311283,
        "MAE":0.04665424391429259
    }}
   
    return metrics  


@app.route("/currencies_pairs/AUD-USD")
def metrics_aud_usd():
    metrics={"AR model": {
        "MSE":1.741659014777564e-05,
        "RMSE":0.004173318840895773,
        "MAE":0.003521097994548128,
        "MAPE":0.5116864664692757
    },
    "ARIMA":{
        "MSE":0.0001111009846518952,
        "RMSE":0.010540445182813447,
        "MAE":0.00947319382991533,
        "MAPE":1.3715546860492003
    },
    "LSTM":{
        "MSE":1.33865493462109e-06,
        "MAE":0.0011187356037103862
    }}
   
    return metrics


@app.route("/currencies_pairs/EUR-CHF")
def metrics_eur_chf():
    metrics={"AR model": {
        "MSE":0.00030440562070237827,
        "RMSE":0.017447223868065034,
        "MAE":0.016289152618739228,
        "MAPE":1.4930270369196879
    },
    "ARIMA":{
        "MSE":"no values",
        "RMSE":"no values",
        "MAE":"no values",
        "MAPE":"no values"
    },
    "LSTM":{
        "MSE":1.5523767718257138e-06,
        "MAE":0.0011856212391520159
    }}
   
    return metrics


@app.route("/currencies_pairs/EUR-JPY")
def metrics_eur_jpy():
    metrics={"AR model": {
        "MSE":0.5147496953878345,
        "RMSE":0.7174605880380013,
        "MAE":0.5765051127688638,
        "MAPE":0.4769771079143303
    },
    "ARIMA":{
        "MSE":1.2365631493203233,
        "RMSE":1.112008610272566,
        "MAE":0.9041489252334723,
        "MAPE":0.7487021520115805
    },
    "LSTM":{
        "MSE":0.006453482102691483,
        "MAE":0.06920059000050059
    }}
   
    return metrics



@app.route("/currencies_pairs/USD-CAD")
def metrics_usd_cad():
    metrics={"AR model": {
        "MSE":0.00015502738279382915,
        "RMSE":0.012450999268887182,
        "MAE":0.010627571555471121,
        "MAPE":0.8092399999695971
    },
    "ARIMA":{
        "MSE":0.00015160379651125056,
        "RMSE":0.012312749348185829,
        "MAE":0.010583139335968918,
        "MAPE":0.8057682571433928
    },
    "LSTM":{
        "MSE":1.219216980426838e-07,
        "MAE":0.00030136238895184036
    }}
   
    return metrics



@app.route("/currencies_pairs/USD-CHF")
def metrics_usd_chf():
    metrics={"AR model": {
        "MSE":0.00019715294563072764,
        "RMSE":0.014041116253016626,
        "MAE":0.012984486520519279,
        "MAPE":1.3240289129443967
    },
    "ARIMA":{
        "MSE":0.00017326048154725095,
        "RMSE":0.01316284473612186,
        "MAE":0.011822151938955892,
        "MAPE":1.2060969870079636
    },
    "LSTM":{
        "MSE":2.127671385084142e-07,
        "MAE":0.00040612818377459684
    }}
   
    return metrics