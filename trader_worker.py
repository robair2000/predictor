import urllib.request
import urllib.error
import os
import time
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import ccxt
from datetime import datetime, timezone

# --- RENDER WEB SERVER (VOOR POORT-BINDING & HEALTH CHECKS) ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass

def run_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()


# --- CONFIGURATIE FUTURES LIVE TRADING & GEOPTIMALISEERDE FILTERS ---
ASSETS = ['BTC', 'ETH', 'LINK']
TIMEFRAMES = ['15m', '1h']
TRADE_CHECK_INTERVAL_SECONDS = 900    # Elke 15 minuten nieuwe trades zoeken
POSITION_CHECK_INTERVAL_SECONDS = 60  # Elke 1 minuut open posities / trailing stops controleren

TRADE_AMOUNT_USDT = 6.0               # Totale gewenste positiewaarde in USDT (~$6 totaal met 10x hefboom)
LEVERAGE = 10                         # Vaste hefboom van 10x

# Basis & Risico Drempels
MIN_DIRECTION_PROBABILITY = 60.0      # Minimale stijgings-/daalwinst kans (60%)
MIN_PREDICTED_RETURN_PCT = 0.30       # Minimale verwachte koersbeweging in %
MIN_RRR = 1.55                        # Minimale Risk/Reward Ratio (Harde filter)
MIN_ATR_PCT = 0.25                    # Minimale volatiliteit (% ATR) (Harde filter)
MAX_ATR_PCT = 5.5                     # Maximale volatiliteit (% ATR) (Harde filter)
MAX_FUNDING_RATE = 0.0005             # Maximaal toegestane funding rate (0.05%)
COOLDOWN_SECONDS = 3600               # 1 uur cooldown na een stop-loss of verlies

# Geavanceerde Quant Parameters
MIN_ORDERBOOK_IMBALANCE = 0.55        # Orderboek druk (minstens 55% steun)
ENABLE_BTC_CORRELATION = True         # Altcoins mogen niet tegen BTC in handelen

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

def pull_latest_states_from_github():
    """Download de nieuwste JSON state-bestanden rechtstreeks van GitHub zonder git nodig te hebben."""
    github_token = os.environ.get('GITHUB_TOKEN', '')
    assets = ['btc', 'eth', 'link']
    timeframes = ['15m', '1h']
    
    files_to_download = [f'model_state_{asset}_{tf}.json' for asset in assets for tf in timeframes]
    
    headers = {}
    if github_token:
        headers['Authorization'] = f'token {github_token}'
        
    for filename in files_to_download:
        url = f'https://raw.githubusercontent.com/robair2000/predictor/main/{filename}'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            if github_token:
                req.add_header('Authorization', f'token {github_token}')
                
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    content = response.read()
                    with open(filename, 'wb') as f:
                        f.write(content)
        except Exception as e:
            pass
            
    print("🔄 Verse JSON-bestanden succesvol opgehaald van GitHub via directe download.")

def manage_open_positions_and_trailing_stops():
    """Beheert open posities, past trailing stops toe en sluit bij tegenwind of trailing stop hit (draait elke minuut)."""
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
        current_atr = current_price * 0.015
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
                symbol_cooldowns[symbol] = time.time()

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
                symbol_cooldowns[symbol] = time.time()

def execute_live_trades_scan():
    """Voert de filtering uit op basis van harde en zachte filters (puntsysteem) en zoekt naar nieuwe trades (draait elke 15 minuten)."""
    pull_latest_states_from_github()
    
    print("--- [Futures Live Trader Worker] Start geoptimaliseerde scoring filtering ---")
    
    try:
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij verbinden met KuCoin Futures: {e}")
        return

    for asset in ASSETS:
        symbol = f"{asset}/USDT:USDT"
        if symbol not in exchange.markets:
            symbol = f"{asset}/USDT"

        # --- CIRCUIT BREAKER / COOLDOWN ---
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
        btc_state_15m = 'model_state_btc_15m.json'

        if os.path.exists(state_file_15m):
            try:
                with open(state_file_15m, 'r', encoding='utf-8') as f:
                    state_15m = json.load(f)
                    
                    predicted_price = state_15m.get('last_prediction', 0.0)
                    prob = state_15m.get('direction_probability', 50.0)
                    
                    ticker = exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    # --- RICHTINGSBEPALING (LONG VS SHORT) ---
                    last_pred = 1 if predicted_price > current_price else 0

                    # --- LIVE ATR & DYNAMIC SL/TP BEREKENING ---
                    current_atr = current_price * 0.015 
                    
                    if last_pred == 1: # Long
                        stop_loss_val = current_price - (1.5 * current_atr)
                        take_profit_val = current_price + (3.0 * current_atr)
                    else: # Short
                        stop_loss_val = current_price + (1.5 * current_atr)
                        take_profit_val = current_price - (3.0 * current_atr)

                    atr_pct = (current_atr / current_price) * 100
                    predicted_return_pct = abs((take_profit_val - current_price) / current_price) * 100

                    # ==========================================
                    # 1. HARDE FILTERS (Poortwachters)
                    # ==========================================
                    if atr_pct < MIN_ATR_PCT or atr_pct > MAX_ATR_PCT:
                        print(f"⚠️ Volatiliteitsfilter weigert {symbol}: ATR {atr_pct:.2f}% valt buiten grenzen.")
                        continue

                    risk = abs(current_price - stop_loss_val)
                    reward = abs(take_profit_val - current_price)
                    rrr = reward / risk if risk > 0 else 0

                    if rrr < MIN_RRR:
                        print(f"⚠️ RRR filter weigert {symbol}: RRR is {rrr:.2f} (minimaal {MIN_RRR})")
                        continue

                    # ==========================================
                    # 2. ZACHTE FILTERS (Puntensysteem / Weging)
                    # ==========================================
                    score = 0
                    max_score = 4

                    # Zacht filter 1: Model Consensus (15m en 1h eensgezind)
                    consensus_ok = True
                    if os.path.exists(state_file_1h):
                        with open(state_file_1h, 'r', encoding='utf-8') as f_1h:
                            state_1h = json.load(f_1h)
                            pred_1h_price = state_1h.get('last_prediction', current_price)
                            pred_1h = 1 if pred_1h_price > current_price else 0
                            if last_pred != pred_1h:
                                consensus_ok = False
                    
                    if consensus_ok:
                        score += 1
                    else:
                        print(f"ℹ️ [Score] 15m en 1h modellen oneens over richting voor {symbol} (-1 punt).")

                    # Zacht filter 2: Bitcoin Macro-Correlatie
                    btc_corr_ok = True
                    if ENABLE_BTC_CORRELATION and asset != 'BTC' and os.path.exists(btc_state_15m):
                        with open(btc_state_15m, 'r', encoding='utf-8') as f_btc:
                            btc_st = json.load(f_btc)
                            btc_pred_price = btc_st.get('last_prediction', current_price)
                            btc_pred = 1 if btc_pred_price > current_price else 0
                            if last_pred == 1 and btc_pred == 0:
                                btc_corr_ok = False
                    
                    if btc_corr_ok:
                        score += 1
                    else:
                        print(f"ℹ️ [Score] Tegen BTC-trend in voor {symbol} (-1 punt).")

                    # Zacht filter 3: Orderboek Imbalance
                    orderbook_ok = True
                    try:
                        orderbook = exchange.fetch_order_book(symbol, limit=20)
                        bids_vol = sum([bid[1] for bid in orderbook.get('bids', [])])
                        asks_vol = sum([ask[1] for ask in orderbook.get('asks', [])])
                        total_vol = bids_vol + asks_vol
                        if total_vol > 0:
                            bid_ratio = bids_vol / total_vol
                            ask_ratio = asks_vol / total_vol
                            if last_pred == 1 and bid_ratio < MIN_ORDERBOOK_IMBALANCE:
                                orderbook_ok = False
                            elif last_pred == 0 and ask_ratio < MIN_ORDERBOOK_IMBALANCE:
                                orderbook_ok = False
                    except Exception:
                        pass

                    if orderbook_ok:
                        score += 1
                    else:
                        print(f"ℹ️ [Score] Onvoldoende orderboek druk voor {symbol} (-1 punt).")

                    # Zacht filter 4: Funding Rate
                    funding_ok = True
                    try:
                        funding_data = exchange.fetch_funding_rate(symbol)
                        current_funding = funding_data.get('fundingRate', 0.0)
                        if current_funding is not None:
                            if last_pred == 1 and current_funding > MAX_FUNDING_RATE:
                                funding_ok = False
                            elif last_pred == 0 and current_funding < -MAX_FUNDING_RATE:
                                funding_ok = False
                    except Exception:
                        pass

                    if funding_ok:
                        score += 1
                    else:
                        print(f"ℹ️ [Score] Ongunstige funding rate voor {symbol} (-1 punt).")

                    MINIMUM_REQUIRED_SCORE = 2  # Minimaal 2 van de 4 punten vereist

                    direction_str = "LONG 📈" if last_pred == 1 else "SHORT 📉"
                    print(f"Beoordeling {symbol} ({direction_str}) -> Score: {score}/{max_score} | Kans: {prob:.1f}%, Move: {predicted_return_pct:.2f}%, RRR: {rrr:.2f}")

                    # --- UITVOERING OP BASIS VAN PUNTEN & DREMPEL ---
                    if prob >= MIN_DIRECTION_PROBABILITY and predicted_return_pct >= MIN_PREDICTED_RETURN_PCT and score >= MINIMUM_REQUIRED_SCORE:
                        # Config: gewenste margin mode (gebruik 'isolated' of 'cross')
                        DESIRED_MARGIN_MODE = os.environ.get('MARGIN_MODE', 'cross')

                        # Probeer margin mode & leverage in te stellen, en alleen dan de order plaatsen
                        margin_and_leverage_ok = False

                        try:
                            has = getattr(exchange, 'has', {}) or {}

                            # 1) Probeer margin mode in te stellen als de exchange dit ondersteunt
                            if has.get('set_margin_mode') or has.get('setMarginMode'):
                                try:
                                    if hasattr(exchange, 'set_margin_mode'):
                                        exchange.set_margin_mode(DESIRED_MARGIN_MODE, symbol)
                                    else:
                                        exchange.setMarginMode(DESIRED_MARGIN_MODE, {'symbol': symbol})
                                    print(f"⚙️ Margin mode ingesteld op {DESIRED_MARGIN_MODE} voor {symbol}")
                                except Exception as e:
                                    print(f"⚠️ Kon margin mode niet instellen op {DESIRED_MARGIN_MODE} voor {symbol}: {e}")
                                    raise
                            else:
                                print("ℹ️ Exchange geeft geen set_margin_mode aan; verifieer margin mode in KuCoin UI indien nodig.")

                            # 2) Nu leverage instellen (probeer unified call, fallback met param)
                            try:
                                exchange.set_leverage(LEVERAGE, symbol)
                            except Exception as e_leverage:
                                try:
                                    exchange.set_leverage(LEVERAGE, symbol, {'marginMode': DESIRED_MARGIN_MODE})
                                except Exception as e2:
                                    print(f"⚠️ Kon hefboom niet instellen voor {symbol}: {e_leverage} | fallback: {e2}")
                                    raise

                            print(f"⚙️ Hefboom succesvol ingesteld op {LEVERAGE}x ({DESIRED_MARGIN_MODE}) voor {symbol}")
                            margin_and_leverage_ok = True

                        except Exception as e:
                            print(f"⛔ Margin/leverage setup mislukt voor {symbol}: {e}")
                            margin_and_leverage_ok = False

                        # Alleen order plaatsen als margin & leverage OK zijn
                        if not margin_and_leverage_ok:
                            print(f"⛔ Trade geannuleerd voor {symbol} omdat margin/leverage niet gegarandeerd zijn.")
                        else:
                            # TRADE_AMOUNT_USDT is direct het totale positiebedrag (6.0 USDT)
                            position_usdt_value = TRADE_AMOUNT_USDT
                            
                            try:
                                market_info = exchange.market(symbol)
                                contract_size = market_info.get('contractSize', 1.0)
                            except Exception:
                                contract_size = 1.0

                            notional_per_contract = current_price * contract_size
                            if notional_per_contract > 0:
                                raw_amount = position_usdt_value / notional_per_contract
                                contract_amount = max(1.0, round(raw_amount))
                            else:
                                contract_amount = 1.0

                            params = {
                                'stopLoss': {'triggerPrice': stop_loss_val},
                                'takeProfit': {'triggerPrice': take_profit_val},
                            }

                            if last_pred == 1:
                                print(f"🔥 Puntsysteem geslaagd ({score}/{max_score})! Sterk LONG signaal voor {symbol} ({contract_amount} contracten)!")
                                order = exchange.create_order(symbol, 'market', 'buy', contract_amount, None, params)
                                print(f"✅ Long order geplaatst: {order.get('id')}")
                            else:
                                print(f"🔥 Puntsysteem geslaagd ({score}/{max_score})! Sterk SHORT signaal voor {symbol} ({contract_amount} contracten)!")
                                order = exchange.create_order(symbol, 'market', 'sell', contract_amount, None, params)
                                print(f"✅ Short order geplaatst: {order.get('id')}")
                    else:
                        print(f"ℹ️ Signaal onder drempelwaarde of onvoldoende score ({score}/{max_score}) voor {symbol}.")

            except Exception as e:
                print(f"Fout bij verwerken filters voor {asset}: {e}")

if __name__ == '__main__':
    print("Futures Live Trader Worker is gestart met automatische contract-filtering (~$6 positie / 10x hefboom cross) (24/7)...")
    last_trade_check_time = 0

    while True:
        current_time = time.time()
        
        # 1. Beheer open posities en trailing stops (draait ALTIJD elke minuut)
        try:
            manage_open_positions_and_trailing_stops()
        except Exception as e:
            print(f"Fout in positiebeheer lus: {e}")
            
        # 2. Zoek naar nieuwe trades via het puntsysteem (draait elke 15 minuten)
        if current_time - last_trade_check_time >= TRADE_CHECK_INTERVAL_SECONDS:
            try:
                execute_live_trades_scan()
            except Exception as e:
                print(f"Fout in trades scan lus: {e}")
            last_trade_check_time = time.time()
            
        # Slaap kort tot de volgende minuut-check
        time.sleep(POSITION_CHECK_INTERVAL_SECONDS)
