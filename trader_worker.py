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


# --- CONFIGURATIE FUTURES LIVE TRADING & TRAILING STOP ---
ASSETS = ['BTC', 'ETH', 'LINK']
TIMEFRAMES = ['15m', '1h', '1d']
CHECK_INTERVAL_SECONDS = 900          # Elke 15 minuten controleren
TRADE_AMOUNT_USDT = 15.0              # Jouw inzet per trade in USDT (marge)
LEVERAGE = 5                          # Vaste hefboom van 5x

# Professionele kwantitatieve filters
MIN_DIRECTION_PROBABILITY = 70.0      # Minimale stijgings-/daalwinst kans (70%)
MIN_PREDICTED_RETURN_PCT = 0.30       # Minimale verwachte koersbeweging in % (0.30% om fees te overwinnen)

# Lokaal geheugen om de piekprices bij te houden voor de Trailing Stop
# Structuur: { "BTC/USDT:USDT": {"peak_price": 85000.0, "sl": 80000.0} }
active_trade_peaks = {}

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

def manage_open_positions_and_trailing_stops():
    """Beheert open posities, past trailing stops toe en sluit bij tegenwind of trailing stop hit."""
    try:
        positions = exchange.fetch_positions()
    except Exception as e:
        print(f"Fout bij ophalen open posities: {e}")
        return

    for pos in positions:
        contracts = float(pos.get('contracts', 0))
        if contracts == 0:
            symbol = pos.get('symbol')
            if symbol in active_trade_peaks:
                del active_trade_peaks[symbol]
            continue

        symbol = pos.get('symbol')
        side = pos.get('side')  # 'long' of 'short'
        entry_price = float(pos.get('entryPrice', 0))
        
        try:
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
        except Exception:
            continue

        # Zoek de bijbehorende state file om de actuele ATR op te halen
        base_asset = symbol.split('/')[0]
        state_file = f'model_state_{base_asset.lower()}_15m.json'
        current_atr = current_price * 0.02  # Fallback ATR van 2%
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    st_data = json.load(f)
                    sl_val = st_data.get('stop_loss', 0)
                    if sl_val > 0:
                        current_atr = abs(entry_price - sl_val) / 1.5
            except Exception:
                pass

        # Initialiseer peak tracking als deze positie nieuw is
        if symbol not in active_trade_peaks:
            if side == 'long':
                initial_sl = entry_price - (1.5 * current_atr)
                active_trade_peaks[symbol] = {"peak_price": current_price, "sl": initial_sl}
            else:
                initial_sl = entry_price + (1.5 * current_atr)
                active_trade_peaks[symbol] = {"peak_price": current_price, "sl": initial_sl}

        trade_info = active_trade_peaks[symbol]

        # Update trailing stop op basis van nieuwe koerspieken
        if side == 'long':
            if current_price > trade_info["peak_price"]:
                trade_info["peak_price"] = current_price
                new_trailing_sl = current_price - (1.5 * current_atr)
                if new_trailing_sl > trade_info["sl"]:
                    trade_info["sl"] = new_trailing_sl
                    print(f"📈 Trailing Stop Verhoogd voor {symbol} naar ${trade_info['sl']:.2f}")

            # Controleer of de trailing stop of harde stop is geraakt
            if current_price <= trade_info["sl"]:
                print(f"🛑 Trailing Stop geraakt voor Long {symbol} op ${current_price}. Positie sluiten...")
                exchange.create_order(symbol, 'market', 'sell', contracts, None, {'reduceOnly': True})
                del active_trade_peaks[symbol]

        elif side == 'short':
            if current_price < trade_info["peak_price"]:
                trade_info["peak_price"] = current_price
                new_trailing_sl = current_price + (1.5 * current_atr)
                if new_trailing_sl < trade_info["sl"]:
                    trade_info["sl"] = new_trailing_sl
                    print(f"📉 Trailing Stop Verlaagd voor {symbol} naar ${trade_info['sl']:.2f}")

            if current_price >= trade_info["sl"]:
                print(f"🛑 Trailing Stop geraakt voor Short {symbol} op ${current_price}. Positie sluiten...")
                exchange.create_order(symbol, 'market', 'buy', contracts, None, {'reduceOnly': True})
                del active_trade_peaks[symbol]

def execute_live_trades():
    print("--- [Futures Live Trader Worker] Start controle op signalen & trailing stops ---")
    
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij verbinden met KuCoin Futures: {e}")
        return

    # Beheer eerst open posities en trailing stops
    manage_open_positions_and_trailing_stops()

    # Controleer op nieuwe signalen
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
                        
                        symbol = f"{asset}/USDT:USDT"
                        if symbol not in exchange.markets:
                            symbol = f"{asset}/USDT"
                        
                        ticker = exchange.fetch_ticker(symbol)
                        current_price = ticker['last']
                        
                        # Bereken de voorspelde procentuele beweging van het model ten opzichte van de huidige koers
                        predicted_return_pct = 0.0
                        if last_pred is not None:
                            # Als we een regressor-waarde hebben opgeslagen, gebruiken we die, anders schatten we op basis van TP/SL verhouding
                            # Hier pakken we de absolute verwachting of schatting
                            # Voor de veiligheid berekenen we het procentuele verschil op basis van de verwachte prijsstap
                            estimated_target = state.get('last_prediction_price', None)
                            if estimated_target and estimated_target > 0:
                                predicted_return_pct = abs((estimated_target - current_price) / current_price) * 100
                            else:
                                # Fallback: Als het model geen losse prijsstap heeft opgeslagen, kijken we naar de afstand tot de take-profit gedeeld door de stap
                                predicted_return_pct = abs((take_profit_val - current_price) / current_price) * 100 / 2.5 # Benadering van 1 ATR stap

                        print(f"Beoordelen {symbol} ({tf}) -> Voorspelling: {last_pred}, Kans: {prob:.1f}%, Verwachte Move: {predicted_return_pct:.2f}%")

                        # CONTROLE OP KANS ÉN MINIMALE RENDEMENTSDREMPEL
                        if prob >= MIN_DIRECTION_PROBABILITY and predicted_return_pct >= MIN_PREDICTED_RETURN_PCT:
                            try:
                                exchange.set_leverage(LEVERAGE, symbol)
                            except Exception:
                                pass

                            position_usdt_value = TRADE_AMOUNT_USDT * LEVERAGE
                            amount = position_usdt_value / current_price

                            if last_pred == 1:
                                print(f"🔥 Sterk LONG signaal met voldoende verwachte marge ({predicted_return_pct:.2f}%) voor {symbol}!")
                                params = {
                                    'stopLoss': {'triggerPrice': stop_loss_val},
                                    'takeProfit': {'triggerPrice': take_profit_val},
                                }
                                order = exchange.create_order(symbol, 'market', 'buy', amount, None, params)
                                print(f"✅ Long futures order succesvol geplaatst: {order.get('id')}")
                                
                            elif last_pred == 0:
                                print(f"🔻 Sterk SHORT signaal met voldoende verwachte marge ({predicted_return_pct:.2f}%) voor {symbol}!")
                                params = {
                                    'stopLoss': {'triggerPrice': stop_loss_val},
                                    'takeProfit': {'triggerPrice': take_profit_val},
                                }
                                order = exchange.create_order(symbol, 'market', 'sell', amount, None, params)
                                print(f"✅ Short futures order succesvol geplaatst: {order.get('id')}")
                        else:
                            print(f"ℹ️ Signaal overgeslagen voor {symbol} ({tf}): Kans ({prob:.1f}%) of verwachte move ({predicted_return_pct:.2f}%) te laag.")
                            
                except Exception as e:
                    print(f"Fout bij lezen van {state_file} of uitvoeren trade: {e}")

if __name__ == '__main__':
    print("Futures Live Trader Worker met Trailing Stop en Rendement-filters is gestart (24/7)...")
    while True:
        execute_live_trades()
        time.sleep(CHECK_INTERVAL_SECONDS)
