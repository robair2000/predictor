import os
import subprocess
from flask import Flask

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Quant Worker Server is actief!", 200

@app.route("/run-worker", methods=["GET", "POST"])
def run_worker():
    try:
        print("Start quant_worker.py...")
        # Voer je bestaande worker script uit
        result = subprocess.run(["python", "quant_worker.py"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Worker succesvol gedraaid!")
            return {"status": "success", "output": result.stdout}, 200
        else:
            print(f"Fout in worker: {result.stderr}")
            return {"status": "error", "message": result.stderr}, 500
            
    except Exception as e:
        return {"status": "error", "message": str(e)}, 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
