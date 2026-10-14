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


# --- CONFIGURATIE LIVE TRADING ---
ASSETS = ['BTC', 'ETH', 'LINK']
TIMEFRAMES = ['15m', '1h', '1d']
CHECK_INTERVAL_SECONDS = 900  # Elke 15 minuten controleren
TRADE_AMOUNT_USDT = 15.0      # Het USDT-bedrag dat je per echte trade wilt inzetten (pas dit aan naar wens)

# Initialiseer KuCoin voor de LIVE markt met je Render Environment Variables
exchange = ccxt.kucoin({
    'apiKey': os.getenv('KUCOIN_API_KEY', ''),
    'secret': os.getenv('KUCOIN_API_SECRET', ''),
    'password': os.getenv('KUCOIN_API_PASSWORD', ''),
    'enableRateLimit': True,
})

def execute_live_trades():
    print("--- [Live Trader Worker] Start controle op signalen ---")
    
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
                        
                        # Haal de huidige marktprijs op om te berekenen hoeveel crypto je moet kopen
                        ticker = exchange.fetch_ticker(symbol)
                        current_price = ticker['last']
                        
                        # Bereken hoeveel base currency je kunt kopen voor jouw TRADE_AMOUNT_USDT
                        amount_to_buy = TRADE_AMOUNT_USDT / current_price

                        if prob >= 70.0 and last_pred == 1:
                            print(f"🔥 Sterk KOOP-signaal voor {symbol}! Echte marktorder plaatsen...")
                            try:
                                # Plaats een echte market buy order op KuCoin
                                order = exchange.create_market_buy_order(symbol, amount_to_buy)
                                print(f"✅ Koop-order succesvol geplaatst: {order['id']}")
                            except Exception as order_err:
                                print(f"❌ Fout bij plaatsen koop-order voor {symbol}: {order_err}")
                            
                        elif prob >= 70.0 and last_pred == 0:
                            print(f"🔻 Sterk VERKOOP-signaal voor {symbol}! Echte marktorder plaatsen...")
                            try:
                                # Plaats een echte market sell order op KuCoin (zorg dat je de munt bezit!)
                                order = exchange.create_market_sell_order(symbol, amount_to_buy)
                                print(f"✅ Verkoop-order succesvol geplaatst: {order['id']}")
                            except Exception as order_err:
                                print(f"❌ Fout bij plaatsen verkoop-order voor {symbol}: {order_err}")
                            
                except Exception as e:
                    print(f"Fout bij lezen van {state_file} of uitvoeren trade: {e}")
            else:
                print(f"Geen state-bestand gevonden voor {asset} ({tf})")

if __name__ == '__main__':
    print("Live Trader Worker is gestart en draait 24/7 met echt geld...")
    while True:
        execute_live_trades()
        time.sleep(CHECK_INTERVAL_SECONDS)
