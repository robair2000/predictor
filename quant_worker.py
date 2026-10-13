import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import json
import os
import subprocess
import time
import gc
import urllib.request
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

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
    'Supertrend_dist', # Vaste basisfeature
    'macro_trend'      # Multitimeframe macro trend context
]

optional_features = [
    'MACD_hist',
    'ADX',
    'CMF',
    'ATR_percentile',
    'regime_score',     # Market Regime (Trend vs Chop)
    'smc_zone_dist',    # Smart Money Concepts / Order Blocks (zonder lek)
    'lorentzian_dist',  # Lorentzian patroonafstand (zonder lek)
    'oi_change',        # Open Interest % verandering (Binance Futures)
    'dist_oi_sma'       # Afstand tot Open Interest SMA
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
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df.set_index('Date', inplace=True)
        return df.reset_index(drop=False)
    except Exception as e:
        print(f"[{ticker}] Fout bij ophalen data: {e}")
        return pd.DataFrame()

def fetch_binance_open_interest(asset):
    """Haalt historische Open Interest data op via Binance Futures API."""
    try:
        symbol = f"{asset}USDT"
        url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period=1h&limit=100"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if isinstance(data, list) and len(data) > 0:
                df_oi = pd.DataFrame(data)
                df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms').dt.tz_localize(None)
                df_oi['sumOpenInterest'] = df_oi['sumOpenInterest'].astype(float)
                df_oi = df_oi.sort_values('timestamp').reset_index(drop=True)
                return df_oi[['timestamp', 'sumOpenInterest']]
    except Exception:
        pass
    return pd.DataFrame()

def merge_price_and_oi(df_prices, df_oi):
    if df_oi.empty:
        df_prices['oi_change'] = 0.0
        df_prices['dist_oi_sma'] = 0.0
        return df_prices

    df_prices = df_prices.sort_values('Date').copy()
    df_prices['Date'] = pd.to_datetime(df_prices['Date']).astype('datetime64[ms]')
    
    df_oi = df_oi.sort_values('timestamp').copy()
    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp']).astype('datetime64[ms]')

    df_merged = pd.merge_asof(
        df_prices, 
        df_oi, 
        left_on='Date', 
        right_on='timestamp', 
        direction='backward'
    )
    
    df_merged['sumOpenInterest'] = df_merged['sumOpenInterest'].ffill().fillna(0.0)
    df_merged['oi_change'] = df_merged['sumOpenInterest'].pct_change().fillna(0.0)
    df_merged['oi_sma'] = df_merged['sumOpenInterest'].rolling(window=20, min_periods=1).mean()
    df_merged['dist_oi_sma'] = ((df_merged['sumOpenInterest'] - df_merged['oi_sma']) / (df_merged['oi_sma'] + 1e-8)).fillna(0.0)
    
    if 'timestamp' in df_merged.columns:
        df_merged.drop(columns=['timestamp'], inplace=True)
    if 'oi_sma' in df_merged.columns:
        df_merged.drop(columns=['oi_sma'], inplace=True)
        
    return df_merged

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
        df_oi = fetch_binance_open_interest(asset)

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

                # 1. Technische Indicatoren & Basisfeatures
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

                # Supersnelle vectoriële Supertrend
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

                # Optionele indicatoren
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

                # 2. Multitimeframe (MTF) Macro Trend
                macro_ema = df['close'].ewm(span=200, adjust=False).mean()
                df['macro_trend'] = (df['close'] - macro_ema) / (macro_ema + 1e-8)

                # 3. Market Regime & Leakage-Free Features
                adx_val = df['ADX'] if 'ADX' in df.columns else dx.rolling(window=14).mean()
                df['regime_score'] = (adx_val / 100.0) * (1.0 - df['atr_ratio'] / (df['atr_ratio'].rolling(20).mean() + 1e-8))

                body_size = (df['close'] - df['open']).abs()
                avg_body = body_size.shift(1).rolling(20).mean()
                is_order_block = body_size.shift(1) > (2.0 * avg_body)
