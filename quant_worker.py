import json
import os
import subprocess
import time
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

assets = ['BTC', 'ETH', 'LINK', 'AAVE']

timeframes = {
    '15m': {'interval': '15m', 'period': '59d', 'horizon': 1},
    '1h': {'interval': '1h', 'period': '730d', 'horizon': 1},
    '4h': {'interval': '1h', 'period': '730d', 'resample': '4h', 'horizon': 1},
    '1d': {'interval': '1d', 'period': 'max', 'horizon': 1},
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

def fetch_data_safely(ticker, tf_info):
    try:
        df = yf.download(ticker, period=tf_info['period'], interval=tf_info['interval'], progress=False)
        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        date_col = 'Datetime' if 'Datetime' in df.columns else ('Date' if 'Date' in df.columns else df.columns[0])
        df = df.rename(columns={
            date_col: 'Date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        if tf_info.get('resample') == '4h':
            df = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            df.dropna(inplace=True)

        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

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
    print("--- Quant Worker gestart ---")
    for asset in assets:
        ticker = f'{asset}-USD'
        for tf_name, tf_info in timeframes.items():
            try:
                state_file = f'model_state_{asset.lower()}_{tf_name}.json'

                df = fetch_data_safely(ticker, tf_info)
                if df is None or df.empty:
                    continue

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
                
                # Geef een DataFrame mee inclusief column names i.p.v. een numpy array
                latest_features = df_clean[features].iloc[[-1]]

                model_xgb = xgb.XGBRegressor(
                    learning_rate=state['learning_rate'],
                    max_depth=state['max_depth'],
                    n_estimators=state['n_estimators'],
                    random_state=42
                )
                model_xgb.fit(X, y)
                pred_xgb = float(model_xgb.predict(latest_features)[0])

                model_lgb = lgb.LGBMRegressor(
                    learning_rate=state['learning_rate'],
                    max_depth=state['max_depth'],
                    n_estimators=state['n_estimators'],
                    random_state=42,
                    verbose=-1
                )
                model_lgb.fit(X, y)
                pred_lgb = float(model_lgb.predict(latest_features)[0])

                model_cat = CatBoostRegressor(
                    learning_rate=state['learning_rate'],
                    depth=state['max_depth'],
                    iterations=state['n_estimators'],
                    random_seed=42,
                    verbose=0
                )
                model_cat.fit(X, y)
                pred_cat = float(model_cat.predict(latest_features)[0])

                next_prediction = (pred_xgb + pred_lgb + pred_cat) / 3
                state['last_prediction'] = next_prediction

                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=4)

                time.sleep(2) # Korte pauze om rate limits te voorkomen
            except Exception:
                pass

    push_to_github()
    print("--- Quant Worker voltooid en gepusht naar GitHub ---")

if __name__ == '__main__':
    main()
