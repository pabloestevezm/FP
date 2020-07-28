from src.app import app
from src.config import PORT
import src.controllers.ta
import src.controllers.companies

app.run("0.0.0.0", PORT, debug=True)