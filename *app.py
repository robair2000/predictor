import streamlit as st
import pandas as df_lib
import numpy as np
import yfinance as yf
import xgboost as xgb
import os

# Pagina configuratie
st.set_page_config(
    page_title="Crypto Quant Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ RV Predictor")
st.markdown("*Real-time machine learning voorspellingen gevoed door een 24/7 achtergrond-worker.*")

# Sidebar: Asset keuze & Tijdinterval (Timescales)
asset_options = {
    'Bitcoin (BTC)': {'yf': 'BTC-USD', 'file': 'model_btc.json'},
    'Ethereum (ETH)': {'yf': 'ETH-USD', 'file': 'model_eth.json'},
    'Chainlink (LINK)': {'yf': 'LINK-USD', 'file': 'model_link.json'},
    'Aave (AAVE)': {'yf': 'AAVE-USD', 'file': 'model_aave.json'},
}

interval_options = {
    '15 Minuten (Micro ML)': {'interval': '15m', 'period': '14d'},
    '1 Uur (Intraday)': {'interval': '1h', 'period': '60d'},
    '1 Dag (Macro Trend)': {'interval': '1d', 'period': '1y'}
}

selected_asset_name = st.sidebar.selectbox("Selecteer Asset", list(asset_options.keys()))
selected_interval_name = st.sidebar.selectbox("Selecteer Tijdinterval", list(interval_options.keys()))

asset_info = asset_options[selected_asset_name]
interval_info = interval_options[selected_interval_name]

# Functie om data op te halen op basis van de gekozen timescale
@st.cache_data(ttl=300)
def load_dashboard_data(ticker, interval, period):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, df_lib.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
        df = df.rename(columns={date_col: 'Date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        df['Date'] = df_lib.to_datetime(df['Date'])
        df = df.sort_values('Date').dropna(subset=['close']).reset_index(drop=True)
        
        # Technische indicatoren
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
        true_range = df_lib.concat([high_low, high_close, low_close], axis=1).max(axis=1)
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
        plus_dm_s = df_lib.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()
        minus_dm_s = df_lib.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()
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
        return df.dropna().reset_index(drop=True)
    except Exception as e:
        return None

features = [
    'close', 'open', 'high', 'low', 'volume', 'VWAP_distance', 'BB_pct_b', 
    'ATR', 'SMA_20', 'EMA_9', 'EMA_21', 'MACD', 'MACD_Signal', 'RSI', 
    'Stoch_K', 'Stoch_D', 'OBV', 'ADX', 'Fib_382', 'Fib_500', 'Fib_618', 
    'Fib_618_distance', 'fear_greed', 'mtf_trend_bullish',
    'RVOL', 'upper_wick', 'lower_wick', 'RSI_momentum', 'MACD_hist_momentum'
]

# Laad de data op basis van gekozen parameters
df = load_dashboard_data(asset_info['yf'], interval_info['interval'], interval_info['period'])

if df is not None and not df.empty:
    current_price = float(df['close'].iloc[-1])
    
    # Laad het opgeslagen achtergrondmodel
    model_file = asset_info['file']
    if os.path.exists(model_file):
        model = xgb.XGBRegressor()
        model.load_model(model_file)
        
        latest_features = df[features].iloc[-1].values.reshape(1, -1)
        predicted_price = model.predict(latest_features)[0]
        pct_diff = ((predicted_price - current_price) / current_price) * 100
        
        # Schone weergave zonder donkere achtergrondblokken
        col1, col2, col3 = st.columns(3)
        col1.metric("Huidige Koers", f"${current_price:,.2f}")
        col2.metric("XGBoost Voorspelling (Bg-Model)", f"${predicted_price:,.2f}", delta=f"{pct_diff:+.2f}%")
        col3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
        
        st.subheader(f"Prijsverloop ({selected_interval_name}) - {selected_asset_name}")
        st.line_chart(df.set_index('Date')['close'])
        
    else:
        st.warning(f"Modelbestand `{model_file}` nog niet gevonden. Wacht even op de volgende achtergrond-cyclus.")
else:
    st.error("Kon geen marktdata ophalen voor deze selectie.")
