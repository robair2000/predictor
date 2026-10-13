import os
import time
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import ccxt

# --- RENDER WEB SERVER (VOOR POORT-BINDING) ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass

def run_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()


# --- CONFIGURATIE TRADING ---
ASSETS = ['BTC', 'ETH', 'LINK']
TIMEFRAMES = ['15m', '1h', '1d']
CHECK_INTERVAL_SECONDS = 900  # Elke 15 minuten controleren

# Initialiseer KuCoin via CCXT (zonder set_sandbox_mode om de crash te voorkomen)
exchange = ccxt.kucoin({
    'apiKey': os.getenv('KUCOIN_SANDBOX_API_KEY', ''),
    'secret': os.getenv('KUCOIN_SANDBOX_SECRET', ''),
    'password': os.getenv('KUCOIN_SANDBOX_PASSWORD', ''),
    'enableRateLimit': True,
})

def execute_paper_trades():
    print("--- [Paper Trader Worker] Start controle op signalen ---")
    
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij verbinden met KuCoin: {e}")
        return

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            state_file = f'model_state_{asset.lower()}_{tf}.json'
            
            if os.path.exists(state_file):
                try:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                        
                        last_pred = state.get('last_prediction')
                        prob = state.get('direction_probability', 50.0)
                        symbol = f"{asset}/USDT"
                        
                        print(f"Beoordelen {symbol} ({tf}) -> Voorspelling: {last_pred}, Kans: {prob:.1f}%")
                        
                        if prob >= 70.0 and last_pred == 1:
                            print(f"🔥 Sterk KOOP-signaal voor {symbol}! Papier-order simuleren...")
                            
                        elif prob >= 70.0 and last_pred == 0:
                            print(f"🔻 Sterk VERKOOP-signaal voor {symbol}! Papier-order simuleren...")
                            
                except Exception as e:
                    print(f"Fout bij lezen van {state_file}: {e}")
            else:
                print(f"Geen state-bestand gevonden voor {asset} ({tf})")

if __name__ == '__main__':
    print("Paper Trader Worker is gestart en draait 24/7...")
    while True:
        execute_paper_trades()
        time.sleep(CHECK_INTERVAL_SECONDS)
