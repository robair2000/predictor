import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Configuratie via GitHub Secrets of directe instelling
WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

asset_options = {
    'Bitcoin (BTC)': {'yf': 'BTC-USD', 'binance': 'BTCUSDT', 'icon': '⚡'},
    'Ethereum (ETH)': {'yf': 'ETH-USD', 'binance': 'ETHUSDT', 'icon': 'Ξ'},
    'Chainlink (LINK)': {'yf': 'LINK-USD', 'binance': 'LINKUSDT', 'icon': '🔗'},
    'Aave (AAVE)': {'yf': 'AAVE-USD', 'binance': 'AAVEUSDT', 'icon': '👻'},
}

def load_data(ticker):
    try:
        df = yf.download(ticker, period="14d", interval="15m", progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df = df.rename(columns={date_col: 'Date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').dropna(subset=['close']).reset_index(drop=True)
        
        # Indicatoren toevoegen
        df['daily_close'] = df['close']
        df['daily_SMA50'] = df['close'].rolling(window=50).mean().bfill()
        
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        upper_band = sma_20 + (std_20 * 2)
        lower_band = sma_20 - (std_20 * 2)
        df['BB_pct_b'] = (df['close'] - lower_band) / (upper_band - lower_band + 1e-8)
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        df['SMA_20'] = sma_20
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['MACD'] = df['EMA_9'] - df['EMA_21']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        rs = gain.rolling(window=14).mean() / loss.rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + rs))
        
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['Stoch_K'] = ((df['close'] - low_14) / (high_14 - low_14 + 1e-8)) * 100
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()
        minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()
        tr_s = true_range.ewm(alpha=1/14, adjust=False).mean()
        plus_di = 100 * (plus_dm_s / (tr_s + 1e-8))
        minus_di = 100 * (minus_dm_s / (tr_s + 1e-8))
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()
        
        rolling_high = df['high'].rolling(window=50).max()
        rolling_low = df['low'].rolling(window=50).min()
        diff_hl = rolling_high - rolling_low
        df['Fib_382'] = rolling_high - (diff_hl * 0.382)
        df['Fib_500'] = rolling_high - (diff_hl * 0.500)
        df['Fib_618'] = rolling_high - (diff_hl * 0.618)
        df['Fib_618_distance'] = (df['close'] - df['Fib_618']) / (df['Fib_618'] + 1e-8)
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['VWAP'] = (typical_price * df['volume']).rolling(window=96 * 2).sum() / (df['volume'].rolling(window=96 * 2).sum() + 1e-8)
        df['VWAP_distance'] = (df['close'] - df['VWAP']) / (df['VWAP'] + 1e-8)
        
        df['mtf_trend_bullish'] = (df['daily_close'] > df['daily_SMA50']).astype(int)
        df['fear_greed'] = 50.0  # Standaard fallback voor worker
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        print(f"Fout bij laden {ticker}: {e}")
        return None

features = [
    'close', 'open', 'high', 'low', 'volume', 'VWAP_distance', 'BB_pct_b', 
    'ATR', 'SMA_20', 'EMA_9', 'EMA_21', 'MACD', 'MACD_Signal', 'RSI', 
    'Stoch_K', 'Stoch_D', 'OBV', 'ADX', 'Fib_382', 'Fib_500', 'Fib_618', 
    'Fib_618_distance', 'fear_greed', 'mtf_trend_bullish'
]

def run_quant_scan():
    print("Start automatische quant scan...")
    reports = []
    
    for name, info in asset_options.items():
        df = load_data(info['yf'])
        if df is None or len(df) < 40:
            continue
            
        df['Target'] = df['close'].shift(-1)
        df_clean = df.dropna().reset_index(drop=True)
        X = df_clean[features]
        y = df_clean['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        curr_p = float(df['close'].iloc[-1])
        latest_data = X.iloc[-1].values.reshape(1, -1)
        pred_p = model.predict(latest_data)[0]
        pct_diff = ((pred_p - curr_p) / curr_p) * 100
        
        is_bull = bool(df['mtf_trend_bullish'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        adx = float(df['ADX'].iloc[-1])
        
        report = f"{info['icon']} **{name}** | Koers: ${curr_p:,.3f} | Doel: ${pred_p:,.3f} ({pct_diff:+.2f}%) | ADX: {adx:.1f}"
        reports.append(report)
        
        # Webhook sturen als er een sterke beweging is (> 0.5% voorspelling en bullish)
        if pct_diff > 0.5 and is_bull and WEBHOOK_URL:
            payload = {
                "content": f"🚨 **QUANT SIGNAL ALERT** 🚨\n{report}\nSetup is sterk bullish!"
            }
            try:
                requests.post(WEBHOOK_URL, json=payload, timeout=5)
            except Exception as e:
                print(f"Webhook fout: {e}")

    print("\n".join(reports))

if __name__ == "__main__":
    run_quant_scan()
