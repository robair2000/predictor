import json
import os
import subprocess
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ccxt
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Initialiseer Binance via CCXT
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

assets = ['BTC', 'ETH', 'LINK', 'AAVE']

timeframes = {
    '15m': {'ccxt_tf': '15m', 'horizon': 1},
    '1h': {'ccxt_tf': '1h', 'horizon': 1},
    '4h': {'ccxt_tf': '4h', 'horizon': 1},
    '1d': {'ccxt_tf': '1d', 'horizon': 1},
}

def load_or_init_state(state_file):
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            return json.load(f)
    else:
        return {
            'learning_rate': 0.05,
            'max_depth': 5,
            'n_estimators': 100,
            'previous_learning_rate': 0.05,
            'previous_max_depth': 5,
            'baseline_error': float('inf'),
            'last_prediction': None,
        }

def evaluate_and_adapt(state, current_price):
    if state['last_prediction'] is not None:
        error = abs(state['last_prediction'] - current_price)
        if error < state['baseline_error']:
            state['baseline_error'] = error
        else:
            state['learning_rate'] = state['previous_learning_rate']
            state['max_depth'] = state['previous_max_depth']

    state['previous_learning_rate'] = state['learning_rate']
    state['previous_max_depth'] = state['max_depth']

    mutation = np.random.choice([-0.005, 0.005], p=[0.5, 0.5])
    state['learning_rate'] = float(
        np.clip(state['learning_rate'] + mutation, 0.01, 0.1)
    )
    return state

features = [
    'close', 'open', 'high', 'low', 'volume',
    'SMA_20', 'EMA_9', 'EMA_21', 'MACD', 'RSI', 'ATR'
]

def fetch_two_years_data(symbol, timeframe):
    since = exchange.parse8601((datetime.utcnow() - timedelta(days=730)).isoformat())
    all_bars = []
    current_since = since

    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=current_since, limit=1000)
            if not bars:
                break
            all_bars.extend(bars)
            current_since = bars[-1][0] + 1
            if bars[-1][0] >= exchange.milliseconds() - (60 * 1000):
                break
            time.sleep(exchange.rateLimit / 1000)
        except Exception:
            break

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

def push_to_github():
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        return
    try:
        repo_url = f'https://{github_token}@github.com/robair2000/predictor.git'
        subprocess.run(['git', 'config', '--global', 'user.email', 'render-worker@bot.com'], check=True)
        subprocess.run(['git', 'config', '--global', 'user.name', 'Render Quant Bot'], check=True)
        subprocess.run(['git', 'add', '*.json'], check=True)

        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True)
        if not status.stdout.strip():
            return

        subprocess.run(['git', 'commit', '-m', 'Auto-update Ensemble models [skip ci]'], check=True)
        branch_res = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True, check=True)
        current_branch = branch_res.stdout.strip() or 'main'
        subprocess.run(['git', 'push', '--force', repo_url, f'HEAD:{current_branch}'], capture_output=True, text=True)
    except Exception:
        pass

def main():
    for asset in assets:
        symbol = f'{asset}/USDT'
        for tf_name, tf_info in timeframes.items():
            try:
                state_file = f'model_state_{asset.lower()}_{tf_name}.json'

                df = fetch_two_years_data(symbol, tf_info['ccxt_tf'])
                if df is None or df.empty:
                    continue

                # Technische indicatoren
                df['SMA_20'] = df['close'].rolling(window=20).mean()
                df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
                df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
                df['MACD'] = df['EMA_9'] - df['EMA_21']

                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                rs = gain.rolling(window=14).mean() / (loss.rolling(window=14).mean() + 1e-8)
                df['RSI'] = 100 - (100 / (1 + rs))

                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['ATR'] = true_range.rolling(window=14).mean()

                df_clean = df.dropna(subset=features + ['close']).reset_index(drop=True)
                if df_clean.empty:
                    continue

                current_price = float(df_clean['close'].iloc[-1])
                state = load_or_init_state(state_file)
                state = evaluate_and_adapt(state, current_price)

                horizon = tf_info['horizon']
                X = df_clean[features].iloc[:-horizon]
                y = df_clean['close'].shift(-horizon).dropna()
                X = X.iloc[: len(y)]
                latest_features = df_clean[features].iloc[-1].values.reshape(1, -1)

                # --- 1. XGBoost Model ---
                model_xgb = xgb.XGBRegressor(
                    learning_rate=state['learning_rate'],
                    max_depth=state['max_depth'],
                    n_estimators=state['n_estimators'],
                    random_state=42
                )
                model_xgb.fit(X, y)
                pred_xgb = float(model_xgb.predict(latest_features)[0])

                # --- 2. LightGBM Model ---
                model_lgb = lgb.LGBMRegressor(
                    learning_rate=state['learning_rate'],
                    max_depth=state['max_depth'],
                    n_estimators=state['n_estimators'],
                    random_state=42,
                    verbose=-1
                )
                model_lgb.fit(X, y)
                pred_lgb = float(model_lgb.predict(latest_features)[0])

                # --- 3. CatBoost Model ---
                model_cat = CatBoostRegressor(
                    learning_rate=state['learning_rate'],
                    depth=state['max_depth'],
                    iterations=state['n_estimators'],
                    random_seed=42,
                    verbose=0
                )
                model_cat.fit(X, y)
                pred_cat = float(model_cat.predict(latest_features)[0])

                # --- Ensemble Consensus (Gemiddelde) ---
                next_prediction = (pred_xgb + pred_lgb + pred_cat) / 3
                state['last_prediction'] = next_prediction

                # Sla state op
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=4)

                time.sleep(1)
            except Exception:
                pass

    push_to_github()

if __name__ == '__main__':
    main()
