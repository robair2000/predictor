from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import xgboost as xgb
import yfinance as yf
from sklearn.model_selection import train_test_split

# --- STREAMLIT PAGINA CONFIGURATIE ---
st.set_page_config(
    page_title='Institutional Multi-Asset Quant & Confluence Dashboard',
    page_icon='⚡',
    layout='wide',
)

st.title('⚡ Institutional Core-Asset Multi-Timeframe Quant Model')
st.markdown(
    'Geoptimaliseerd & lichtgewicht quant dashboard — **Core Assets (BTC, ETH, LINK, AAVE)**, '
    '**Bliksemsnelle Caching**, **Adaptieve Logica** en **Crashvrij**.'
)

# --- ASSET & TIJDSCHAAL CONFIGURATIE (Alleen Core Assets) ---
asset_options = {
    'Bitcoin (BTC)': {'yf': 'BTC-USD', 'binance': 'BTCUSDT', 'icon': '⚡'},
    'Ethereum (ETH)': {'yf': 'ETH-USD', 'binance': 'ETHUSDT', 'icon': 'Ξ'},
    'Chainlink (LINK)': {'yf': 'LINK-USD', 'binance': 'LINKUSDT', 'icon': '🔗'},
    'Aave (AAVE)': {'yf': 'AAVE-USD', 'binance': 'AAVEUSDT', 'icon': '👻'},
}

timeframe_options = {
    '15 Minuten (15m)': {'interval': '15m', 'period': '14d', 'bars_per_day': 96, 'type': 'micro'},
    '30 Minuten (30m)': {'interval': '30m', 'period': '30d', 'bars_per_day': 48, 'type': 'micro'},
    '1 Uur (1h)': {'interval': '1h', 'period': '60d', 'bars_per_day': 24, 'type': 'micro'},
    '4 Uur (4h)': {'interval': '1h', 'period': '180d', 'bars_per_day': 6, 'type': 'micro'},
    '1 Dag (1d)': {'interval': '1d', 'period': '1y', 'bars_per_day': 1, 'type': 'macro'},
    '1 Week (1wk)': {'interval': '1wk', 'period': 'max', 'bars_per_day': 1, 'type': 'macro'},
    '1 Maand (1mo)': {'interval': '1mo', 'period': 'max', 'bars_per_day': 1, 'type': 'macro'},
}

st.sidebar.header('⚙️ Dashboard Instellingen')
selected_asset_name = st.sidebar.selectbox('Selecteer Hoofdasset', list(asset_options.keys()))
asset_info = asset_options[selected_asset_name]

selected_tf_name = st.sidebar.selectbox('Selecteer Tijdschaal (Candle)', list(timeframe_options.keys()))
tf_info = timeframe_options[selected_tf_name]

test_size = st.sidebar.slider('Test / Train Splitsing (%)', 10, 40, 20, step=5)
n_estimators = st.sidebar.slider('XGBoost Bomen (n_estimators)', 50, 200, 100)
max_depth = st.sidebar.slider('Maximale Boomdiepte (max_depth)', 3, 8, 4)

st.sidebar.markdown('---')
st.sidebar.subheader('🤖 Notificatie / Webhook (Optioneel)')
webhook_url = st.sidebar.text_input('Discord/Telegram Webhook URL')
send_alert = st.sidebar.button('Stuur Signaal Test')


# --- DATA FUNCTIES MET LICHTGEWICHT CACHING ---
@st.cache_data(ttl=600, show_spinner=False)
def load_market_data(ticker, interval, period):
  try:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
      return None

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={date_col: 'Date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').dropna(subset=['close']).reset_index(drop=True)

    if len(df) > 1000:
      df = df.tail(1000).reset_index(drop=True)

    df['daily_close'] = df['close']
    df['daily_SMA50'] = df['close'].rolling(window=50).mean().bfill()
    df = df.dropna().reset_index(drop=True)
    return df if not df.empty else None
  except Exception:
    return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_realtime_binance_price(binance_symbol):
  try:
    url = f'https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}'
    res = requests.get(url, timeout=2).json()
    return float(res['price'])
  except Exception:
    return None


def add_advanced_indicators(df, bars_per_day):
  vwap_window = max(12, int(bars_per_day * 2))
  typical_price = (df['high'] + df['low'] + df['close']) / 3
  df['VWAP'] = (typical_price * df['volume']).rolling(window=vwap_window).sum() / (df['volume'].rolling(window=vwap_window).sum() + 1e-8)
  df['VWAP_distance'] = (df['close'] - df['VWAP']) / (df['VWAP'] + 1e-8)

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
