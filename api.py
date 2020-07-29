from src.app import app
from src.config import PORT

app.run("0.0.0.0", PORT, debug=True)