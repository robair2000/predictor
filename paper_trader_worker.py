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
        # Stilhouden om output limieten op Render te vermijden
        pass

def run_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

# Start de webserver direct in een daemon-thread op de achtergrond
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()


# --- CONFIGURATIE TRADING ---
ASSETS = ['BTC', 'ETH', 'LINK']
TIMEFRAMES = ['15m', '1h', '1d']
CHECK_INTERVAL_SECONDS = 900  # Elke 15 minuten controleren

# Stel KuCoin in via CCXT voor Sandbox (Testnet)
exchange = ccxt.kucoin({
    'apiKey': os.getenv('KUCOIN_SANDBOX_API_KEY', 'JOUW_SANDBOX_API_KEY'),
    'secret': os.getenv('KUCOIN_SANDBOX_SECRET', 'JOUW_SANDBOX_SECRET'),
    'password': os.getenv('KUCOIN_SANDBOX_PASSWORD', 'JOUW_SANDBOX_PASSPHRASE'), # KuCoin vereist altijd een passphrase
    'enableRateLimit': True,
})

# Schakel de sandbox / testnet modus in voor 100% veiligheid met nepgeld
exchange.set_sandbox_mode(True)

def execute_paper_trades():
    print("--- [Paper Trader Worker] Start controle op signalen via KuCoin Sandbox ---")
    
    try:
        # Laad de markten in van de exchange
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij verbinden met KuCoin Sandbox: {e}")
        return

    for asset in ASSETS:
        for tf in TIMEFRAMES:
            state_file = f'model_state_{asset.lower()}_{tf}.json'
            
            if os.path.exists(state_file):
                try:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                        
                        last_pred = state.get('last_prediction')  # Bijv. 1 (Buy) of 0 (Sell)
                        prob = state.get('direction_probability', 50.0) # Bijv. 75.5%
                        symbol = f"{asset}/USDT"
                        
                        print(f"Beoordelen {symbol} ({tf}) -> Voorspelling: {last_pred}, Kans: {prob:.1f}%")
                        
                        # Voorbeeld logica: Alleen handelen bij een sterke kans (> 70%)
                        if prob >= 70.0 and last_pred == 1:
                            print(f"🔥 Sterk KOOP-signaal voor {symbol}! Papier-order plaatsen op KuCoin Sandbox...")
                            
                        elif prob >= 70.0 and last_pred == 0:
                            print(f"🔻 Sterk VERKOOP-signaal voor {symbol}! Papier-order sluiten/shorten...")
                            
                except Exception as e:
                    print(f"Fout bij lezen van {state_file}: {e}")
            else:
                print(f"Geen state-bestand gevonden voor {asset} ({tf})")

if __name__ == '__main__':
    print("Paper Trader Worker (KuCoin Sandbox) is gestart en draait 24/7...")
    while True:
        execute_paper_trades()
        # Wacht tot de volgende ronde (bijv. 15 minuten)
        time.sleep(CHECK_INTERVAL_SECONDS)
