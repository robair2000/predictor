import os
import threading
from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Quant Worker Server is actief!"

@app.route("/run-worker")
def trigger_worker():
    try:
        from quant_worker import main as run_quant_worker
        worker_thread = threading.Thread(target=run_quant_worker)
        worker_thread.start()
    except Exception as e:
        print(f"Fout bij starten worker: {e}")
    
    # Altijd een minimale respons van 2 letters teruggeven
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
