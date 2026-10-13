import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import json
import os
import subprocess
import time
import gc
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import logging
import warnings

# Onderdruk alle standaard waarschuwingen en externe bibliotheek logs
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.ERROR)
logging.getLogger('xgboost').setLevel(logging.ERROR)
logging.getLogger('lightgbm').setLevel(logging.ERROR)

assets = ['BTC', 'ETH', 'LINK']

# Tijdsframes met geoptimaliseerde periodes om RAM-pieken op Render te voorkomen
timeframes = {
    '15m': {'interval': '15m', 'period': '59d', 'horizon': 1, 'min_interval_seconds': 14 * 60},
    '1h': {'interval': '1h', 'period': '365d', 'horizon': 1, 'min_interval_seconds': 55 * 60},
    '1d': {'interval': '1d', 'period': '730d', 'horizon': 1, 'min_interval_seconds': 23 * 3600},
    '1w': {'interval': '1wk', 'period': '730d', 'horizon': 1, 'min_interval_seconds': 6 * 86400},
    '1m': {'interval': '1mo', 'period': 'max', 'horizon': 1, 'min_interval_seconds': 25 * 86400},
}

base_features = [
    'return_1',
    'dist_sma20',
    'dist_sma50',
    'dist_ema9',
    'dist_ema21',
    'RSI',
    'atr_ratio',
    'bb_position',
    'OBV_slope',
    'VWAP_dist',
    'Supertrend_dist'  # Vaste basisfeature
]

# Uitbreiding van optionele features met alle geavanceerde institutionele wensen tegelijk
optional_features = [
    'MACD_hist',
    'ADX',
    'CMF',
    'ATR_percentile',
    'regime_score',     # 1. Market Regime (Trend vs Chop)
    'smc_zone_dist',    # 2. Smart Money Concepts / Order Blocks afstand
    'lorentzian_dist'   # 3. KNN / Lorentzian-achtige patroonafstand
]

def load_or_init_state(state_file):
    models = ['xgb', 'lgb', 'cat', 'rf']
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
            state.setdefault('search_direction', 1)
            state.setdefault('lookback_window', 150)
            state.setdefault('active_features', base_features.copy())
            state.setdefault('last_trained_timestamp', 0)
            
            if 'model_weights' not in state:
                state['model_weights'] = {}
            if 'model_errors' not in state:
                state['model_errors'] = {}
            
            initial_weight = 1.0 / len(models)
            for m in models:
                state['model_weights'].setdefault(m, initial_weight)
                state['model_errors'].setdefault(m, float('inf'))
            return state
    else:
        initial_weight = 1.0 / len(models)
        return {
            'learning_rate': 0.05,
            'max_depth': 5,
            'n_estimators': 100,
            'baseline_error': float('inf'),
            'last_prediction': None,
            'search_direction': 1,
            'lookback_window': 150,
            'active_features': base_features.copy(),
            'last_trained_timestamp': 0,
            'model_weights': {m: initial_weight for m in models},
            'model_errors': {m: float('inf') for m in models},
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

    if np.random.rand() < 0.3:
        available_opts = [f for f in optional_features if f not in state['active_features']]
        if available_opts and len(state['active_features']) < (len(base_features) + len(optional_features)):
            new_feat = np.random.choice(available_opts)
            state['active_features'].append(new_feat)
        elif len(state['active_features']) > len(base_features):
            removable_opts = [f for f in state['active_features'] if f in optional_features]
            if removable_opts:
                removed = np.random.choice(removable_opts)
                state['active_features'] = [f for f in state['active_features'] if f != removed]

    return state

def update_dynamic_weights(state, preds, current_price):
    errors = {k: abs(v - current_price) for k, v in preds.items()}
    state['model_errors'] = errors

    inv_errors = {k: (1.0 / (v + 1e-8)) for k, v in errors.items()}
    total_inv = sum(inv_errors.values())
    
    state['model_weights'] = {k: float(v / total_inv) for k, v in inv_errors.items()}
    return state

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
        print("[GitHub] Waarschuwing: GITHUB_TOKEN ontbreekt in Render environment variables!")
        return
    try:
        repo_url = f'https://{github_token}@github.com/robair2000/predictor.git'
        subprocess.run(['git', 'config', '--global', 'user.email', 'render-worker@bot.com'], check=True)
        subprocess.run(['git', 'config', '--global', 'user.name', 'Render Quant Bot'], check=True)
        subprocess.run(['git', 'add', '*.json'], check=True)

        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=True)
        if not status.stdout.strip():
            print("[GitHub] Geen gewijzigde .json bestanden gevonden om te pushen.")
            return

        subprocess.run(['git', 'commit', '-m', 'Auto-update Smart Evolutionary Ensemble [skip ci]'], check=True)
        branch_res = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True, check=True)
        current_branch = branch_res.stdout.strip() or 'main'
        
        subprocess.run(['git', 'push', '--force', repo_url, f'HEAD:{current_branch}'], capture_output=True, check=True)
        print(f"[GitHub] Succesvol gepusht naar branch {current_branch}!")
    except Exception as e:
        print(f"[GitHub] Fout opgetreden tijdens GitHub push: {e}")

def main():
    print("--- Smart Evolutionary Ensemble Quant Worker gestart ---")
    current_time = time.time()
    
    for asset in assets:
        ticker = f'{asset}-USD'
        for tf_name, tf_info in timeframes.items():
            state_file = f'model_state_{asset.lower()}_{tf_name}.json'
            try:
                state = load_or_init_state(state_file)
                
                last_trained = state.get('last_trained_timestamp', 0)
                if tf_name != '15m' and (current_time - last_trained < tf_info['min_interval_seconds']):
                    continue

                df = fetch_data_safely(ticker, tf_info)
                if df is None or df.empty or len(df) < 60:
                    continue

                # Standaard indicatoren
                df['SMA_20'] = df['close'].rolling(window=20).mean()
                df['SMA_50'] = df['close'].rolling(window=50).mean()
                df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
                df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()

                df['return_1'] = df['close'].pct_change()
                df['dist_sma20'] = (df['close'] - df['SMA_20']) / df['SMA_20']
                df['dist_sma50'] = (df['close'] - df['SMA_50']) / df['SMA_50']
                df['dist_ema9'] = (df['close'] - df['EMA_9']) / df['EMA_9']
                df['dist_ema21'] = (df['close'] - df['EMA_21']) / df['EMA_21']

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
                df['atr_ratio'] = df['ATR'] / df['close']

                bb_std = df['close'].rolling(window=20).std()
                bb_upper = df['SMA_20'] + (bb_std * 2)
                bb_lower = df['SMA_20'] - (bb_std * 2)
                df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)

                df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
                df['OBV_slope'] = df['OBV'].pct_change()

                typical_price = (df['high'] + df['low'] + df['close']) / 3
                vwap = (df['volume'] * typical_price).cumsum() / (df['volume'].cumsum() + 1e-8)
                df['VWAP_dist'] = (df['close'] - vwap) / vwap

                # Supersnelle vectoriële Supertrend berekening
                hl2 = (df['high'] + df['low']) / 2
                atr3 = 3.0 * df['ATR']
                basic_upper = hl2 + atr3
                basic_lower = hl2 - atr3
                
                upper_val = basic_upper.to_numpy(copy=True)
                lower_val = basic_lower.to_numpy(copy=True)
                close_val = df['close'].to_numpy(copy=True)
                
                f_upper = basic_upper.to_numpy(copy=True)
                f_lower = basic_lower.to_numpy(copy=True)
                t_arr = np.ones(len(df), dtype=int)
                
                upper_val.flags.writeable = True
                lower_val.flags.writeable = True
                close_val.flags.writeable = True
                f_upper.flags.writeable = True
                f_lower.flags.writeable = True
                t_arr.flags.writeable = True
                
                for i in range(1, len(df)):
                    if np.isnan(upper_val[i]) or np.isnan(f_upper[i-1]):
                        continue
                        
                    if upper_val[i] < f_upper[i-1] or close_val[i-1] > f_upper[i-1]:
                        f_upper[i] = upper_val[i]
                    else:
                        f_upper[i] = f_upper[i-1]
                        
                    if lower_val[i] > f_lower[i-1] or close_val[i-1] < f_lower[i-1]:
                        f_lower[i] = lower_val[i]
                    else:
                        f_lower[i] = f_lower[i-1]
                        
                    if close_val[i] > f_upper[i-1]:
                        t_arr[i] = 1
                    elif close_val[i] < f_lower[i-1]:
                        t_arr[i] = -1
                    else:
                        t_arr[i] = t_arr[i-1]
                        
                supertrend = np.where(t_arr == 1, f_lower, f_upper)
                df['Supertrend_dist'] = (df['close'] - supertrend) / (df['close'] + 1e-8)

                # Standaard optionele indicatoren
                ema_12 = df['close'].ewm(span=12, adjust=False).mean()
                ema_26 = df['close'].ewm(span=26, adjust=False).mean()
                macd_line = ema_12 - ema_26
                macd_signal = macd_line.ewm(span=9, adjust=False).mean()
                df['MACD_hist'] = macd_line - macd_signal

                plus_dm = df['high'].diff()
                minus_dm = df['low'].diff()
                plus_dm = np.where((plus_dm > 0) & (plus_dm > minus_dm), plus_dm, 0.0)
                minus_dm = np.where((minus_dm > 0) & (minus_dm > plus_dm), minus_dm, 0.0)
                tr14 = true_range.rolling(window=14).sum()
                plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).sum() / (tr14 + 1e-8))
                minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).sum() / (tr14 + 1e-8))
                dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
                df['ADX'] = dx.rolling(window=14).mean()

                mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / ((df['high'] - df['low']) + 1e-8)
                mf_volume = mf_multiplier * df['volume']
                df['CMF'] = mf_volume.rolling(window=20).sum() / (df['volume'].rolling(window=20).sum() + 1e-8)

                df['ATR_percentile'] = df['ATR'].rolling(window=100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

                # ==========================================
                # GEAVANCEERDE MODulaire FEATURES TOEVOEGEN
                # ==========================================
                
                # 1. Market Regime Detection (Trend vs Range)
                adx_val = df['ADX'] if 'ADX' in df.columns else dx.rolling(window=14).mean()
                df['regime_score'] = (adx_val / 100.0) * (1.0 - df['atr_ratio'] / (df['atr_ratio'].rolling(20).mean() + 1e-8))

                # 2. Smart Money Concepts (SMC) - Afstand tot recente grote wick-zones (Order Blocks)
                body_size = (df['close'] - df['open']).abs()
                avg_body = body_size.rolling(20).mean()
                is_order_block = body_size > (2.0 * avg_body)
                recent_ob_price = np.where(is_order_block, (df['high'] + df['low']) / 2, np.nan)
                df['smc_zone_dist'] = (df['close'] - pd.Series(recent_ob_price).ffill()) / (df['close'] + 1e-8)
                df['smc_zone_dist'] = df['smc_zone_dist'].fillna(0)

                # 3. Lorentzian / KNN-achtige afstand tot lokale toppen en bodems
                rolling_max = df['high'].rolling(10).max()
                rolling_min = df['low'].rolling(10).min()
                mid_range = (rolling_max + rolling_min) / 2
                df['lorentzian_dist'] = (df['close'] - mid_range) / (rolling_max - rolling_min + 1e-8)

                # ==========================================

                current_price = float(df['close'].iloc[-1])
                current_atr = float(df['ATR'].iloc[-1]) if not np.isnan(df['ATR'].iloc[-1]) else (current_price * 0.02)

                state = evaluate_and_adapt(state, current_price)
                state['last_trained_timestamp'] = current_time

                current_features = state['active_features']
                df_clean = df.dropna(subset=current_features + ['close']).reset_index(drop=True)
                if df_clean.empty:
                    continue

                lookback = state['lookback_window']
                df_train = df_clean.iloc[-lookback:] if len(df_clean) > lookback else df_clean

                horizon = tf_info['horizon']
                X = df_train[current_features].iloc[:-horizon]
                y_reg = df_train['close'].shift(-horizon).dropna()
                X = X.iloc[: len(y_reg)]
                
                y_class = (df_train['close'].shift(-horizon) > df_train['close']).astype(int).iloc[:-horizon]
                y_class = y_class.iloc[: len(X)]

                latest_features = df_clean[current_features].iloc[[-1]]

                # Modellen trainen
                model_xgb = xgb.XGBRegressor(learning_rate=state['learning_rate'], max_depth=state['max_depth'], n_estimators=state['n_estimators'], random_state=42)
                model_xgb.fit(X, y_reg)
                pred_xgb = float(model_xgb.predict(latest_features)[0])

                model_lgb = lgb.LGBMRegressor(learning_rate=state['learning_rate'], max_depth=state['max_depth'], n_estimators=state['n_estimators'], random_state=42, verbose=-1)
                model_lgb.fit(X, y_reg)
                pred_lgb = float(model_lgb.predict(latest_features)[0])

                model_cat = CatBoostRegressor(learning_rate=state['learning_rate'], depth=state['max_depth'], iterations=state['n_estimators'], random_seed=42, verbose=0)
                model_cat.fit(X, y_reg)
                pred_cat = float(model_cat.predict(latest_features)[0])

                model_rf = RandomForestRegressor(n_estimators=50, max_depth=state['max_depth'], random_state=42, n_jobs=-1)
                model_rf.fit(X, y_reg)
                pred_rf = float(model_rf.predict(latest_features)[0])

                preds = {
                    'xgb': pred_xgb,
                    'lgb': pred_lgb,
                    'cat': pred_cat,
                    'rf': pred_rf
                }

                state = update_dynamic_weights(state, preds, current_price)
                weights = state['model_weights']

                next_prediction = sum(preds[m] * weights[m] for m in preds)
                state['last_prediction'] = next_prediction

                try:
                    clf_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42, eval_metric='logloss')
                    clf_xgb.fit(X, y_class)
                    prob = clf_xgb.predict_proba(latest_features)[0][1]
                    state['direction_probability'] = float(prob * 100)
                except Exception:
                    state['direction_probability'] = 50.0

                if next_prediction > current_price:
                    state['stop_loss'] = float(current_price - (1.5 * current_atr))
                    state['take_profit'] = float(current_price + (2.5 * current_atr))
                else:
                    state['stop_loss'] = float(current_price + (1.5 * current_atr))
                    state['take_profit'] = float(current_price - (2.5 * current_atr))

                with open(state_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=4)

                # Geheugen direct opruimen per iteratie
                gc.collect()

                time.sleep(1)
            except Exception as e:
                print(f"Fout bij {asset} {tf_name}: {e}")

    push_to_github()
    print("--- Smart Evolutionary Ensemble Quant Worker voltooid ---")

if __name__ == '__main__':
    main()
