import os
import threading
from flask import Flask
from quant_worker import main as run_quant_worker

app = Flask(__name__)

@app.route("/")
def home():
    return "Quant Worker Server is actief!"

@app.route("/run-worker")
def trigger_worker():
    # Start de worker op de achtergrond zodat de HTTP-respons direct teruggaat
    worker_thread = threading.Thread(target=run_quant_worker)
    worker_thread.start()
    
    # Geef een minimale respons van slechts 2 letters terug om "output too large" te voorkomen
    return "OK", 200

if __name__ == "__main__":
    # Haal de poort op die Render toewijst, of val terug op poort 10000 voor lokaal testen
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
