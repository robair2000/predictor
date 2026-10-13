import json
import os

assets = ['BTC', 'ETH', 'LINK']
timeframes = ['15m', '1h', '1d']

def check_predictions():
    print("--- Papierhandel / Signalentest gestart ---")
    for asset in assets:
        for tf in timeframes:
            state_file = f'model_state_{asset.lower()}_{tf}.json'
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    last_pred = state.get('last_prediction')
                    prob = state.get('direction_probability', 50.0)
                    sl = state.get('stop_loss')
                    tp = state.get('take_profit')
                    
                    print(asset, tf, "-> Voorspelling:", last_pred, "| Kans om te stijgen:", f"{prob:.1f}%")
            else:
                print(f"Geen state-bestand gevonden voor {asset} ({tf})")

if __name__ == '__main__':
    check_predictions()
