import threading
from flask import Flask
from quant_worker import main as run_quant_worker  # Pas dit aan als je import anders heet

app = Flask(__name__)


@app.route("/")
def home():
  return "Quant Worker Server is actief!"


@app.route("/run-worker")
def trigger_worker():
  # Start de worker in een achtergrond-thread zodat de HTTP-response direct teruggaat
  worker_thread = threading.Thread(target=run_quant_worker)
  worker_thread.start()

  return (
      "Worker is gestart op de achtergrond en wordt verwerkt.",
      200,
  )


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=10000)
