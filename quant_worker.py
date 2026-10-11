import os
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb

def run_ml_training():
    print("Start automatische achtergrond machine learning training & model opslag...")

    # Lijst van assets die de worker automatisch bijwerkt
    assets = {
        'BTC': 'BTC-USD',
        'ETH': 'ETH-USD',
        'LINK': 'LINK-USD',
        'AAVE': 'AAVE-USD'
    }

    # De complete features lijst inclusief de nieuwe micro-features
    features = [
        'close', 'open', 'high', 'low', 'volume', 'VWAP_distance', 'BB_pct_b', 
        'ATR', 'SMA_20', 'EMA_9', 'EMA_21', 'MACD', 'MACD_Signal', 'RSI', 
        'Stoch_K', 'Stoch_D', 'OBV', 'ADX', 'Fib_382', 'Fib_500', 'Fib_618', 
        'Fib_618_distance', 'fear_greed', 'mtf_trend_bullish',
        'RVOL', 'upper_wick', 'lower_wick', 'RSI_momentum', 'MACD_hist_momentum'
    ]

    for symbol, ticker in assets.items():
        print(f"Bezig met trainen voor {symbol} ({ticker})...")
        try:
            # Data ophalen (15m interval)
            df = yf.download(ticker, period="14d", interval="15m", progress=False)
            if df is None or df.empty:
                print(f"Geen data gevonden voor {symbol}")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df = df.reset_index()
            date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
            df = df.rename(columns={date_col: 'Date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').dropna(subset=['close']).reset_index(drop=True)

            # Technische indicatoren berekenen
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
            df['fear_greed'] = 50.0

            # --- Toegevoegde Micro-Features ---
            df['RVOL'] = df['volume'] / (df['volume'].rolling(window=20).mean() + 1e-8)
            
            body_size = (df['close'] - df['open']).abs() + 1e-8
            df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / body_size
            df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / body_size
            
            df['RSI_momentum'] = df['RSI'].diff()
            macd_histogram = df['MACD'] - df['MACD_Signal']
            df['MACD_hist_momentum'] = macd_histogram.diff()

            # Opschonen van NaN waarden
            df_clean = df.dropna().reset_index(drop=True)
            if len(df_clean) < 50:
                print(f"Te weinig data na opschonen voor {symbol}")
                continue

            # Train en target variabelen
            X = df_clean[features]
            y = df_clean['close'].shift(-1).dropna()
            X = X.iloc[:-1] # Lijn uit met y

            # XGBoost Model trainen
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
            model.fit(X, y)

            # Opslaan als JSON modelbestand
            filename = f"model_{symbol.lower()}.json"
            model.save_model(filename)
            print(f"Model succesvol getraind en opgeslagen als {filename}")

        except Exception as e:
            print(f"Fout bij verwerken van {symbol}: {e}")

if __name__ == "__main__":
    run_ml_training()
