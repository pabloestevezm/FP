
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