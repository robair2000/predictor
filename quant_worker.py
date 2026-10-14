import os
import sys

# --- PERMANENTE PYTHON-LEVEL OUTPUT ONDERDRUKKING ---
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# --- EXTREME OUTPUT & WARNING SUPPRESSION (DIRECT BIJ DE START) ---
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# --- HARDCORE OS-NIVEAU OUTPUT ONDERDRUKKING ---
def suppress_output():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
    except Exception:
        pass

suppress_output()

# Vooraf declareren van main() zodat de server hem direct kan aanroepen op de achtergrond
def main():
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

                adx_val = df['ADX'] if 'ADX' in df.columns else dx.rolling(window=14).mean()
                df['regime_score'] = (adx_val / 100.0) * (1.0 - df['atr_ratio'] / (df['atr_ratio'].rolling(20).mean() + 1e-8))

                body_size = (df['close'] - df['open']).abs()
                avg_body = body_size.rolling(20).mean()
                is_order_block = body_size > (2.0 * avg_body)
                recent_ob_price = np.where(is_order_block, (df['high'] + df['low']) / 2, np.nan)
                df['smc_zone_dist'] = (df['close'] - pd.Series(recent_ob_price).ffill()) / (df['close'] + 1e-8)
                df['smc_zone_dist'] = df['smc_zone_dist'].fillna(0)

                rolling_max = df['high'].rolling(10).max()
                rolling_min = df['low'].rolling(10).min()
                mid_range = (rolling_max + rolling_min) / 2
                df['lorentzian_dist'] = (df['close'] - mid_range) / (rolling_max - rolling_min + 1e-8)

                current_price = float(df['close'].iloc[-1])
                current_atr = float(df['ATR'].iloc[-1]) if not np.isnan(df['ATR'].iloc[-1]) else (current_price * 0.02)

                current_regime = get_market_regime(df)
                state['current_regime'] = current_regime

                if 'regime_features' in state and current_regime in state['regime_features']:
                    state['active_features'] = state['regime_features'][current_regime]
                else:
                    state['active_features'] = base_features.copy()

                state = evaluate_and_adapt(state, current_price, current_regime)
                state['last_trained_timestamp'] = current_time

                current_features = state['active_features']
                df_clean = df.dropna(subset=current_features + ['close']).reset_index(drop=True)
                if df_clean.empty
