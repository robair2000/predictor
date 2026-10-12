import warnings
warnings.filterwarnings('ignore', category=UserWarning)
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
    '1d': {'interval': '1d', 'period': '730d', 'horizon': 1},
    '1w': {'interval': '1wk', 'period': '730d', 'horizon': 1},
    '1m': {'interval': '1mo', 'period': 'max', 'horizon': 1},
}

def load_or_init_state(state_file):
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
            # Zorg dat nieuwe velden bestaan voor oudere state bestanden
            state.setdefault('search_direction', 1)
            state.setdefault('lookback_window', 150)
            state.setdefault('model_weights', {'xgb': 0.33, 'lgb': 0.33, 'cat': 0.34})
            state.setdefault('model_errors', {'xgb': float('inf'), 'lgb': float('inf'), 'cat': float('inf')})
            return state
    else:
        return {
            'learning_rate': 0.05,
            'max_depth': 5,
            'n_estimators': 100,
            'baseline_error': float('inf'),
            'last_prediction': None,
            'search_direction': 1,
            'lookback_window': 150,
            'model_weights': {'xgb': 0.33, 'lgb': 0.33, 'cat': 0.34},
            'model_errors': {'xgb': float('inf'), 'lgb': float('inf'), 'cat': float('inf')},
            'direction_probability': 50.0,
            'stop_loss': 0.0,
            'take_profit': 0.0
        }

def evaluate_and_adapt(state, current_price):
    if state['last_prediction'] is not None:
        error = abs(state['last_prediction'] - current_price)
        if error < state['baseline_error']:
            state['baseline_error'] = error
        else:
            state['search_direction'] *= -1
            state['baseline_error'] = error

    step = 0.005 * state['search_direction']
    state['learning_rate'] = float(np.clip(state['learning_rate'] + step, 0.01, 0.15))
    
    window_mutation = np.random.choice([-10, 10])
    state['lookback_window'] = int(np.clip(state['lookback_window'] + window_mutation, 60, 300))
    return state

def update_dynamic_weights(state, pred_xgb, pred_lgb, pred_cat, current_price):
    """Berekent fouten per model en past de weging dynamisch aan."""
    errors = {
        'xgb': abs(pred_xgb - current_price),
        'lgb': abs(pred_lgb - current_price),
        'cat': abs(pred_cat - current_price)
    }
    state['model_errors'] = errors

    # Inverse error weging: model met de kleinste fout krijgt het grootste gewicht
    inv_errors = {k: (1.0 / (v + 1e-8)) for k, v in errors.items()}
    total_inv = sum(inv_errors.values())
    
    state['model_weights'] = {k: float(v / total_inv) for k, v in inv_errors.items()}
    return state

features = [
    'close', 'open', 'high', 'low', 'volume',
    'SMA_20', 'EMA_9', 'EMA_21', 'MACD', 'RSI', 'ATR',
    'BB_Upper', 'BB_Lower', 'OBV', 'VWAP'
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
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"[{ticker}] Fout bij ophalen data: {e}")
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

        subprocess.run(['git', 'commit', '-m', 'Auto-update Advanced Institutional Ensemble [skip ci]'], check=True)
        branch_res = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True, check=True)
        current_branch = branch_res.stdout.strip() or 'main'
        subprocess.run(['git', 'push', '--force', repo_url, f'HEAD:{current_branch}'], capture_output=True, check=True)
    except Exception:
        pass

def main():
    print("--- Institutionele Quant Worker gestart ---")
    for asset in assets:
        ticker = f'{asset}-USD'
        for tf_name, tf_info in timeframes.items():
            state_file = f'model_state_{asset.lower()}_{tf_name}.json'
            try:
                df = fetch_data_safely(ticker, tf_info)
                if df is None or df.empty or len(df) < 35:
                    continue

                # Geavanceerde Feature Engineering
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

                # Bollinger Bands
                bb_std = df['close'].rolling(window=20).std()
                df['BB_Upper'] = df['SMA_20'] + (bb_std * 2)
                df['BB_Lower'] = df['SMA_20'] - (bb_std * 2)

                # OBV & VWAP
                df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                df['VWAP'] = (df['volume'] * typical_price).cumsum() / (df['volume'].cumsum() + 1e-8)

                df_clean = df.dropna(subset=features + ['close']).reset_index(drop=True)
                if df_clean.empty:
                    continue

                current_price = float(df_clean['close'].iloc[-1])
                current_atr = float(df_clean['ATR'].iloc[-1]) if not np.isnan(df_clean['ATR'].iloc[-1]) else (current_price * 0.02)
                
                state = load_or_init_state(state_file)
                state = evaluate_and_adapt(state, current_price)

                lookback = state['lookback_window']
                df_train = df_clean.iloc[-lookback:] if len(df_clean) > lookback else df_clean

                horizon = tf_info['horizon']
                X = df_train[features].iloc[:-horizon]
                y_reg = df_train['close'].shift(-horizon).dropna()
                X = X.iloc[: len(y_reg)]
                
                # Classificatie target: 1 als koers stijgt, 0 als koers daalt
                y_class = (df_train['close'].shift(-horizon) > df_train['close']).astype(int).iloc[:-horizon]
                y_class = y_class.iloc[: len(X)]

                latest_features = df_clean[features].iloc[[-1]]

                # 1. Regressiemodellen (Prijsvoorspelling)
                model_xgb = xgb.XGBRegressor(learning_rate=state['learning_rate'], max_depth=state['max_depth'], n_estimators=state['n_estimators'], random_state=42)
                model_xgb.fit(X, y_reg)
                pred_xgb = float(model_xgb.predict(latest_features)[0])

                model_lgb = lgb.LGBMRegressor(learning_rate=state['learning_rate'], max_depth=state['max_depth'], n_estimators=state['n_estimators'], random_state=42, verbose=-1)
                model_lgb.fit(X, y_reg)
                pred_lgb = float(model_lgb.predict(latest_features)[0])

                model_cat = CatBoostRegressor(learning_rate=state['learning_rate'], depth=state['max_depth'], iterations=state['n_estimators'], random_seed=42, verbose=0)
                model_cat.fit(X, y_reg)
                pred_cat = float(model_cat.predict(latest_features)[0])

                # Dynamische weging bijwerken
                state = update_dynamic_weights(state, pred_xgb, pred_lgb, pred_cat, current_price)
                weights = state['model_weights']

                # Gewogen ensemble prijsvoorspelling
                next_prediction = (pred_xgb * weights['xgb']) + (pred_lgb * weights['lgb']) + (pred_cat * weights['cat'])
                state['last_prediction'] = next_prediction

                # 2. Classificatiemodel (Richtingskans / Confidence)
                try:
                    clf_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42, eval_metric='logloss')
                    clf_xgb.fit(X, y_class)
                    prob = clf_xgb.predict_proba(latest_features)[0][1] # Kans op stijging (1)
                    state['direction_probability'] = float(prob * 100)
                except Exception:
                    state['direction_probability'] = 50.0

                # 3. Automatisch Risicobeheer (Stop-Loss & Take-Profit gebaseerd op ATR)
                if next_prediction > current_price:
                    # Long setup
                    state['stop_loss'] = float(current_price - (1.5 * current_atr))
                    state['take_profit'] = float(current_price + (2.5 * current_atr))
                else:
                    # Short setup
                    state['stop_loss'] = float(current_price + (1.5 * current_atr))
                    state['take_profit'] = float(current_price - (2.5 * current_atr))

                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=4)

                time.sleep(1)
            except Exception as e:
                print(f"Fout bij {asset} {tf_name}: {e}")

    push_to_github()
    print("--- Institutionele Quant Worker voltooid ---")

if __name__ == '__main__':
    main()
