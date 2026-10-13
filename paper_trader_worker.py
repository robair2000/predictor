import os
import time
import json
import ccxt

# --- CONFIGURATIE ---
ASSETS = ['BTC', 'ETH', 'LINK']
TIMEFRAMES = ['15m', '1h', '1d']
CHECK_INTERVAL_SECONDS = 900  # Elke 15 minuten controleren

# Stel hier je exchange in (bijv. Binance in Testnet/Sandbox modus)
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_TESTNET_API_KEY', 'JOUW_TESTNET_API_KEY'),
    'secret': os.getenv('BINANCE_TESTNET_SECRET', 'JOUW_TESTNET_SECRET'),
    'enableRateLimit': True,
})

# Schakel sandbox / testnet in voor 100% veiligheid met nepgeld
exchange.set_sandbox_mode(True)

def execute_paper_trades():
    print("--- [Paper Trader Worker] Start controle op signalen ---")
    
    try:
        # Optioneel: laad actuele marktprijzen
        exchange.load_markets()
    except Exception as e:
        print(f"Fout bij verbinden met exchange: {e}")
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
                            print(f"🔥 Sterk KOOP-signaal voor {symbol}! Papier-order plaatsen...")
                            # Hier plaats je eventueel je CCXT testnet order logica:
                            # order = exchange.create_market_buy_order(symbol, hoeveelheid)
                            
                        elif prob >= 70.0 and last_pred == 0:
                            print(f"🔻 Sterk VERKOOP-signaal voor {symbol}! Papier-order sluiten/shorten...")
                            
                except Exception as e:
                    print(f"Fout bij lezen van {state_file}: {e}")
            else:
                print(f"Geen state-bestand gevonden voor {asset} ({tf})")

if __name__ == '__main__':
    print("Paper Trader Worker is gestart en draait 24/7...")
    while True:
        execute_paper_trades()
        # Wacht tot de volgende ronde (bijv. 15 minuten)
        time.sleep(CHECK_INTERVAL_SECONDS)
