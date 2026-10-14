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


# --- CONFIGURATIE FUTURES LIVE TRADING ---
ASSETS = ['BTC', 'ETH', 'LINK']
TIMEFRAMES = ['15m', '1h', '1d']
CHECK_INTERVAL_SECONDS = 900  # Elke 15 minuten controleren
TRADE_AMOUNT_USDT = 15.0      # Jouw inzet per trade in USDT (marge)
LEVERAGE = 5                  # Vaste hefboom van 5x

# Initialiseer KuCoin voor FUTURES (Swap) met je Render Environment Variables
exchange = ccxt.kucoin({
    'apiKey': os.getenv('KUCOIN_API_KEY', ''),
    'secret': os.getenv('KUCOIN_API_SECRET', ''),
    'password': os.getenv('KUCOIN_API_PASSWORD', ''),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',  # Cruciaal voor Futures / Perpetual swaps
    }
})

def execute_live_trades():
    print("--- [Futures Live Trader Worker] Start controle op signalen met 5x leverage ---")
    
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij verbinden met KuCoin Futures: {e}")
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
                        stop_loss_val = state.get('stop_loss', 0.0)
                        take_profit_val = state.get('take_profit', 0.0)
                        
                        # KuCoin Futures notatie in CCXT (bijv. BTC/USDT:USDT)
                        symbol = f"{asset}/USDT:USDT"
                        if symbol not in exchange.markets:
                            symbol = f"{asset}/USDT"
                        
                        print(f"Beoordelen {symbol} ({tf}) -> Voorspelling: {last_pred}, Kans: {prob:.1f}%")
                        
                        # Haal de huidige marktprijs op
                        ticker = exchange.fetch_ticker(symbol)
                        current_price = ticker['last']
                        
                        # Stel de hefboom in op 5x voor dit handelspaar
                        try:
                            exchange.set_leverage(LEVERAGE, symbol)
                        except Exception as lev_err:
                            pass  # Milde melding negeren als de leverage al op 5x staat

                        # Bereken positiegrootte op basis van de inzet vermenigvuldigd met 5x hefboom
                        position_usdt_value = TRADE_AMOUNT_USDT * LEVERAGE
                        amount = position_usdt_value / current_price

                        if prob >= 70.0 and last_pred == 1:
                            print(f"🔥 Sterk LONG (KOOP) signaal voor {symbol} met 5x leverage! Order + TP/SL plaatsen...")
                            try:
                                params = {
                                    'stopLoss': {'triggerPrice': stop_loss_val},
                                    'takeProfit': {'triggerPrice': take_profit_val},
                                }
                                order = exchange.create_order(symbol, 'market', 'buy', amount, None, params)
                                print(f"✅ Long futures order succesvol geplaatst: {order.get('id')}")
                            except Exception as order_err:
                                print(f"❌ Fout bij plaatsen long order voor {symbol}: {order_err}")
                            
                        elif prob >= 70.0 and last_pred == 0:
                            print(f"🔻 Sterk SHORT (VERKOOP) signaal voor {symbol} met 5x leverage! Order + TP/SL plaatsen...")
                            try:
                                params = {
                                    'stopLoss': {'triggerPrice': stop_loss_val},
                                    'takeProfit': {'triggerPrice': take_profit_val},
                                }
                                order = exchange.create_order(symbol, 'market', 'sell', amount, None, params)
                                print(f"✅ Short futures order succesvol geplaatst: {order.get('id')}")
                            except Exception as order_err:
                                print(f"❌ Fout bij plaatsen short order voor {symbol}: {order_err}")
                            
                except Exception as e:
                    print(f"Fout bij lezen van {state_file} of uitvoeren futures trade: {e}")
            else:
                print(f"Geen state-bestand gevonden voor {asset} ({tf})")

if __name__ == '__main__':
    print("Futures Live Trader Worker is gestart en draait 24/7 met echt geld, 5x leverage, TP en SL...")
    while True:
        execute_live_trades()
        time.sleep(CHECK_INTERVAL_SECONDS)
