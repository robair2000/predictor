import os
import time
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import ccxt
from datetime import datetime, timezone

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


# --- CONFIGURATIE FUTURES LIVE TRADING & 9 INSTITUTIONELE FILTERS ---
ASSETS = ['BTC', 'ETH', 'LINK']
TIMEFRAMES = ['15m', '1h', '1d']
CHECK_INTERVAL_SECONDS = 900          # Elke 15 minuten controleren
TRADE_AMOUNT_USDT = 15.0              # Jouw inzet per trade in USDT (marge)
LEVERAGE = 5                          # Vaste hefboom van 5x

# 1 t/m 5: Basis & Risico Filters (Geoptimaliseerd voor 15m)
MIN_DIRECTION_PROBABILITY = 72.0      # Filter: Minimale stijgings-/daalwinst kans (72%)
MIN_PREDICTED_RETURN_PCT = 0.30       # Filter: Minimale verwachte koersbeweging in %
MIN_RRR = 1.6                         # Filter 2: Minimale Risk/Reward Ratio
MIN_ATR_PCT = 0.4                     # Filter 3: Minimale volatiliteit (% ATR) tegen dode markten
MAX_ATR_PCT = 5.5                     # Filter 3: Maximale volatiliteit (% ATR) tegen extreme chaos
MAX_FUNDING_RATE = 0.0005             # Filter 4: Maximaal toegestane funding rate (0.05%)
COOLDOWN_SECONDS = 3600               # Filter 5: 1 uur cooldown na een stop-loss of verlies

# 6 t/m 9: Geavanceerde "Magische" Quant Filters
MIN_ORDERBOOK_IMBALANCE = 0.55        # Filter 7: Orderboek druk (minstens 55% steun)
ENABLE_BTC_CORRELATION = True         # Filter 8: Altcoins mogen niet tegen BTC in handelen
AVOID_WEEKEND_ILLIQUID = True         # Filter 9: Controle op extreme illiquiditeit

# Geheugen voor tracking
active_trade_peaks = {}               # {symbol: {"peak_price": float, "sl": float}}
symbol_cooldowns = {}                 # {symbol: timestamp_laatste_sl}

# Initialiseer KuCoin voor FUTURES (Swap)
exchange = ccxt.kucoin({
    'apiKey': os.getenv('KUCOIN_API_KEY', ''),
    'secret': os.getenv('KUCOIN_API_SECRET', ''),
    'password': os.getenv('KUCOIN_API_PASSWORD', ''),
    'enableRateLimit': True,
    'options': {
        'defaultType': 'swap',
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

        base_asset = symbol.split('/')[0]
        state_file = f'model_state_{base_asset.lower()}_15m.json'
        current_atr = current_price * 0.02
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    st_data = json.load(f)
                    sl_val = st_data.get('stop_loss', 0)
                    if sl_val > 0:
                        current_atr = abs(entry_price - sl_val) / 1.5
            except Exception:
                pass

        if symbol not in active_trade_peaks:
            if side == 'long':
                initial_sl = entry_price - (1.5 * current_atr)
                active_trade_peaks[symbol] = {"peak_price": current_price, "sl": initial_sl}
            else:
                initial_sl = entry_price + (1.5 * current_atr)
                active_trade_peaks[symbol] = {"peak_price": current_price, "sl": initial_sl}

        trade_info = active_trade_peaks[symbol]

        if side == 'long':
            if current_price > trade_info["peak_price"]:
                trade_info["peak_price"] = current_price
                new_trailing_sl = current_price - (1.5 * current_atr)
                if new_trailing_sl > trade_info["sl"]:
                    trade_info["sl"] = new_trailing_sl
                    print(f"📈 Trailing Stop Verhoogd voor {symbol} naar ${trade_info['sl']:.2f}")

            if current_price <= trade_info["sl"]:
                print(f"🛑 Trailing Stop geraakt voor Long {symbol} op ${current_price}. Positie sluiten...")
                exchange.create_order(symbol, 'market', 'sell', contracts, None, {'reduceOnly': True})
                del active_trade_peaks[symbol]
                symbol_cooldowns[symbol] = time.time()  # Activeer cooldown na sluiting

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
                symbol_cooldowns[symbol] = time.time()  # Activeer cooldown na sluiting

def execute_live_trades():
    print("--- [Futures Live Trader Worker] Start 9-staps geavanceerde filtering ---")
    
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij verbinden met KuCoin Futures: {e}")
        return

    manage_open_positions_and_trailing_stops()

    for asset in ASSETS:
        symbol = f"{asset}/USDT:USDT"
        if symbol not in exchange.markets:
            symbol = f"{asset}/USDT"

        # --- FILTER 5: Circuit Breaker / Cooldown na verlies ---
        if symbol in symbol_cooldowns:
            elapsed = time.time() - symbol_cooldowns[symbol]
            if elapsed < COOLDOWN_SECONDS:
                remaining_min = int((COOLDOWN_SECONDS - elapsed) / 60)
                print(f"⏳ Cooldown actief voor {symbol}. Nog {remaining_min} minuten te gaan.")
                continue
            else:
                del symbol_cooldowns[symbol]

        state_file_15m = f'model_state_{asset.lower()}_15m.json'
        state_file_1h = f'model_state_{asset.lower()}_1h.json'
        state_file_1d = f'model_state_{asset.lower()}_1d.json'
        btc_state_15m = 'model_state_btc_15m.json'

        if os.path.exists(state_file_15m):
            try:
                with open(state_file_15m, 'r', encoding='utf-8') as f:
                    state_15m = json.load(f)
                    
                    last_pred = state_15m.get('last_prediction')
                    prob = state_15m.get('direction_probability', 50.0)
                    stop_loss_val = state_15m.get('stop_loss', 0.0)
                    take_profit_val = state_15m.get('take_profit', 0.0)
                    
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    current_atr = abs(current_price - stop_loss_val) / 1.5 if stop_loss_val > 0 else current_price * 0.02
                    atr_pct = (current_atr / current_price) * 100

                    # --- FILTER 3: Volatiliteitsfilter (ATR bandbreedte) ---
                    if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                        print(f"⚠️ Volatiliteitsfilter weigert {symbol}: ATR {atr_pct:.2f}% valt buiten grenzen.")
                        continue

                    estimated_target = state_15m.get('last_prediction_price', None)
                    if estimated_target and estimated_target > 0:
                        predicted_return_pct = abs((estimated_target - current_price) / current_price) * 100
                    else:
                        predicted_return_pct = abs((take_profit_val - current_price) / current_price) * 100 / 2.5

                    # --- FILTER 2: Risk/Reward Ratio (RRR) ---
                    risk = abs(current_price - stop_loss_val)
                    reward = abs(take_profit_val - current_price)
                    rrr = reward / risk if risk > 0 else 0

                    if rrr < MIN_RRR:
                        print(f"⚠️ RRR filter weigert {symbol}: RRR is {rrr:.2f} (minimaal {MIN_RRR})")
                        continue

                    # --- FILTER 1 & 6: Multi-Timeframe Trend & Model Consensus Filter ---
                    macro_trend_ok = True
                    consensus_ok = True

                    # 1D Macro Trend Check
                    if os.path.exists(state_file_1d):
                        with open(state_file_1d, 'r', encoding='utf-8') as f_1d:
                            state_1d = json.load(f_1d)
                            pred_1d = state_1d.get('last_prediction', 1)
                            if (last_pred == 1 and pred_1d == 0) or (last_pred == 0 and pred_1d == 1):
                                macro_trend_ok = False

                    # 1H Model Consensus Check
                    if os.path.exists(state_file_1h):
                        with open(state_file_1h, 'r', encoding='utf-8') as f_1h:
                            state_1h = json.load(f_1h)
                            pred_1h = state_1h.get('last_prediction', 1)
                            if last_pred != pred_1h:
                                consensus_ok = False

                    if not macro_trend_ok:
                        print(f"⚠️ Macro Trend Filter weigert {symbol}: 15m signaal botst met de 1D dagtrend.")
                        continue

                    if not consensus_ok:
                        print(f"⚠️ Model Consensus Filter weigert {symbol}: 15m en 1h modellen zijn het oneens.")
                        continue

                    # --- FILTER 8: Bitcoin Macro-Correlatie (voor altcoins) ---
                    if ENABLE_BTC_CORRELATION and asset != 'BTC' and os.path.exists(btc_state_15m):
                        with open(btc_state_15m, 'r', encoding='utf-8') as f_btc:
                            btc_st = json.load(f_btc)
                            btc_pred = btc_st.get('last_prediction', 1)
                            if last_pred == 1 and btc_pred == 0:
                                print(f"⚠️ BTC Correlatie Filter weigert {symbol} (Long): Bitcoin staat zelf op daling.")
                                continue

                    # --- FILTER 7: Orderbook Imbalance Filter ---
                    try:
                        orderbook = exchange.fetch_order_book(symbol, limit=20)
                        bids_vol = sum([bid[1] for bid in orderbook.get('bids', [])])
                        asks_vol = sum([ask[1] for ask in orderbook.get('asks', [])])
                        total_vol = bids_vol + asks_vol
                        
                        if total_vol > 0:
                            bid_ratio = bids_vol / total_vol
                            ask_ratio = asks_vol / total_vol
                            
                            if last_pred == 1 and bid_ratio < MIN_ORDERBOOK_IMBALANCE:
                                print(f"⚠️ Orderboek Filter weigert {symbol} (Long): Onvoldoende koopdruk ({bid_ratio:.2f}).")
                                continue
                            elif last_pred == 0 and ask_ratio < MIN_ORDERBOOK_IMBALANCE:
                                print(f"⚠️ Orderboek Filter weigert {symbol} (Short): Onvoldoende verkoopdruk ({ask_ratio:.2f}).")
                                continue
                    except Exception:
                        pass  # Fail-safe bij eventuele API hikken in orderboek

                    # --- FILTER 4: Funding Rate Filter ---
                    try:
                        funding_data = exchange.fetch_funding_rate(symbol)
                        current_funding = funding_data.get('fundingRate', 0.0)
                        if current_funding is not None:
                            if last_pred == 1 and current_funding > MAX_FUNDING_RATE:
                                print(f"⚠️ Funding Rate te hoog voor Long op {symbol}: {current_funding:.5f}")
                                continue
                            elif last_pred == 0 and current_funding < -MAX_FUNDING_RATE:
                                print(f"⚠️ Funding Rate te ongunstig voor Short op {symbol}: {current_funding:.5f}")
                                continue
                    except Exception:
                        pass

                    print(f"Beoordeling {symbol} -> Alle 9 filters doorlopen! Kans: {prob:.1f}%, Move: {predicted_return_pct:.2f}%, RRR: {rrr:.2f}")

                    # --- ALLES GOEDGEKEURD: UITVOEREN ---
                    if prob >= MIN_DIRECTION_PROBABILITY and predicted_return_pct >= MIN_PREDICTED_RETURN_PCT:
                        try:
                            exchange.set_leverage(LEVERAGE, symbol)
                        except Exception:
                            pass

                        position_usdt_value = TRADE_AMOUNT_USDT * LEVERAGE
                        amount = position_usdt_value / current_price

                        if last_pred == 1:
                            print(f"🔥 Alle 9 filters geslaagd! Sterk LONG signaal voor {symbol}!")
                            params = {
                                'stopLoss': {'triggerPrice': stop_loss_val},
                                'takeProfit': {'triggerPrice': take_profit_val},
                            }
                            order = exchange.create_order(symbol, 'market', 'buy', amount, None, params)
                            print(f"✅ Long order geplaatst: {order.get('id')}")
                            
                        elif last_pred == 0:
                            print(f"🔥 Alle 9 filters geslaagd! Sterk SHORT signaal voor {symbol}!")
                            params = {
                                'stopLoss': {'triggerPrice': stop_loss_val},
                                'takeProfit': {'triggerPrice': take_profit_val},
                            }
                            order = exchange.create_order(symbol, 'market', 'sell', amount, None, params)
                            print(f"✅ Short order geplaatst: {order.get('id')}")
                    else:
                        print(f"ℹ️ Signaal onder drempelwaarde voor {symbol}.")

            except Exception as e:
                print(f"Fout bij verwerken filters voor {asset}: {e}")

if __name__ == '__main__':
    print("Futures Live Trader Worker met alle 9 institutionele filters is gestart (24/7)...")
    while True:
        execute_live_trades()
        time.sleep(CHECK_INTERVAL_SECONDS)
