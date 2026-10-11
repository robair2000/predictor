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
    page_title='Multi-Asset Multi-Timeframe Quant Dashboard',
    page_icon='⚡',
    layout='wide',
)

st.title('⚡ Multi-Asset Multi-Timeframe Quant Model')
st.markdown(
    'Dit professionele dashboard combineert **flexibele orderflow**, **Multi-Timeframe'
    ' Trend Filter**, **ATR Risk Management** en **Backtesting** voor een'
    ' uitgebreide suite van crypto-assets — 100% feitelijke data.'
)

# --- SIDEBAR CONFIGURATIE ---
st.sidebar.header('⚙️ Asset & Tijdschaal Instellingen')

asset_options = {
    'Chainlink (LINK)': {'yf': 'LINK-USD', 'binance': 'LINKUSDT', 'icon': '🔗'},
    'Ethereum (ETH)': {'yf': 'ETH-USD', 'binance': 'ETHUSDT', 'icon': 'Ξ'},
    'Bitcoin (BTC)': {'yf': 'BTC-USD', 'binance': 'BTCUSDT', 'icon': '⚡'},
    'PAX Gold (PAXG)': {'yf': 'PAXG-USD', 'binance': 'PAXGUSDT', 'icon': '🟡'},
    'Aave (AAVE)': {'yf': 'AAVE-USD', 'binance': 'AAVEUSDT', 'icon': '👻'},
    'Arbitrum (ARB)': {'yf': 'ARB-USD', 'binance': 'ARBUSDT', 'icon': '🔵'},
    'Avalanche (AVAX)': {'yf': 'AVAX-USD', 'binance': 'AVAXUSDT', 'icon': '🔺'},
    'Canton (CC)': {'yf': 'CC-USD', 'binance': 'CCUSDT', 'icon': '🌐'},
}

selected_asset_name = st.sidebar.selectbox(
    'Selecteer Asset', list(asset_options.keys())
)
asset_info = asset_options[selected_asset_name]

# Tijdschaal mapping voor Yahoo Finance
timeframe_options = {
    '15 Minuten (15m)': {'interval': '15m', 'period': '30d', 'bars_per_day': 96},
    '30 Minuten (30m)': {'interval': '30m', 'period': '60d', 'bars_per_day': 48},
    '1 Uur (1h)': {'interval': '1h', 'period': '730d', 'bars_per_day': 24},
    '4 Uur (4h)': {'interval': '1h', 'period': '730d', 'bars_per_day': 6},
    '1 Dag (1d)': {'interval': '1d', 'period': 'max', 'bars_per_day': 1},
    '1 Week (1wk)': {'interval': '1wk', 'period': 'max', 'bars_per_day': 1},
    '1 Maand (1mo)': {'interval': '1mo', 'period': 'max', 'bars_per_day': 1},
}

selected_tf_name = st.sidebar.selectbox(
    'Selecteer Tijdschaal (Candle)', list(timeframe_options.keys())
)
tf_info = timeframe_options[selected_tf_name]

test_size = st.sidebar.slider('Test / Train Splitsing (%)', 10, 40, 20, step=5)
n_estimators = st.sidebar.slider('XGBoost Bomen (n_estimators)', 50, 300, 150)
max_depth = st.sidebar.slider('Maximale Boomdiepte (max_depth)', 3, 10, 4)

st.sidebar.markdown('---')
st.sidebar.subheader('🤖 Notificatie / Webhook (Optioneel)')
webhook_url = st.sidebar.text_input('Discord/Telegram Webhook URL')
send_alert = st.sidebar.button('Stuur Signaal Test')


# --- DATA OPHAAL (Gekozen Tijdschaal + Daily Trend) ---
@st.cache_data(ttl=300)
def load_market_data(ticker, interval, period):
  try:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df is None or df.empty:
      return None
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(
        columns={
            date_col: 'Date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
        }
    )
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').dropna(subset=['close']).reset_index(drop=True)

    df_daily = yf.download(ticker, period='max', interval='1d', progress=False)
    if isinstance(df_daily.columns, pd.MultiIndex):
      df_daily.columns = df_daily.columns.get_level_values(0)
    df_daily = df_daily.reset_index()
    d_col = 'Datetime' if 'Datetime' in df_daily.columns else 'Date'
    df_daily = df_daily.rename(columns={d_col: 'Date', 'Close': 'daily_close'})
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily['daily_SMA50'] = df_daily['daily_close'].rolling(window=50).mean()

    df['Date_only'] = df['Date'].dt.date
    df_daily['Date_only'] = df_daily['Date'].dt.date
    df = pd.merge(
        df,
        df_daily[['Date_only', 'daily_close', 'daily_SMA50']],
        on='Date_only',
        how='left',
    )
    df = df.drop(columns=['Date_only']).ffill().dropna().reset_index(drop=True)

    return df
  except Exception as e:
    st.error(f'Fout bij ophalen data voor {ticker}: {e}')
    return None


def fetch_realtime_binance_price(binance_symbol):
  try:
    url = f'https://api.binance.com/api/v3/ticker/price?symbol={binance_symbol}'
    res = requests.get(url, timeout=3).json()
    return float(res['price'])
  except Exception:
    return None


def add_advanced_indicators(df, bars_per_day):
  vwap_window = max(24, int(bars_per_day * 4))
  
  typical_price = (df['high'] + df['low'] + df['close']) / 3
  df['VWAP'] = (typical_price * df['volume']).rolling(window=vwap_window).sum() / (
      df['volume'].rolling(window=vwap_window).sum() + 1e-8
  )
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
  df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
  df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
  df['MACD'] = df['EMA_9'] - df['EMA_21']
  df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

  delta = df['close'].diff()
  gain = delta.where(delta > 0, 0.0)
  loss = -delta.where(delta < 0, 0.0)
  rs = gain.rolling(window=14).mean() / loss.rolling(window=14).mean()
  df['RSI'] = 100 - (100 / (1 + rs))

  df['mtf_trend_bullish'] = (df['daily_close'] > df['daily_SMA50']).astype(int)
  return df


@st.cache_data(ttl=300)
def fetch_crypto_fear_greed():
  try:
    res = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5).json()
    return float(res['data'][0]['value'])
  except Exception:
    return 50.0


# --- DATA LADEN VOOR GEKOZEN ASSET & TIJDSCHAAL ---
with st.spinner(f'Laden van {selected_tf_name} data voor {selected_asset_name}...'):
  df = load_market_data(asset_info['yf'], tf_info['interval'], tf_info['period'])
  if df is None or df.empty:
    st.error(f'Kon geen data laden voor {selected_asset_name} op tijdschaal {selected_tf_name}. Controleer of de ticker correct is.')
    st.stop()

  df = add_advanced_indicators(df, tf_info['bars_per_day'])
  df['fear_greed'] = fetch_crypto_fear_greed()
  df['Target'] = df['close'].shift(-1)
  df = df.dropna().reset_index(drop=True)

features = [
    'close',
    'open',
    'high',
    'low',
    'volume',
    'VWAP_distance',
    'BB_pct_b',
    'ATR',
    'SMA_20',
    'EMA_9',
    'EMA_21',
    'MACD',
    'MACD_Signal',
    'RSI',
    'fear_greed',
    'mtf_trend_bullish',
]

X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100.0, shuffle=False
)

# --- MODEL TRAINING ---
model = xgb.XGBRegressor(
    n_estimators=n_estimators, max_depth=max_depth, random_state=42
)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# --- BACKTESTING SIMULATIE OP TESTSET ---
test_preds = model.predict(X_test)
test_actuals = y_test.values
test_closes = X_test['close'].values

pred_direction = np.sign(test_preds - test_closes)
actual_direction = np.sign(test_actuals - test_closes)
directional_acc = np.mean(pred_direction == actual_direction) * 100

# --- DASHBOARD UI ---
realtime_price = fetch_realtime_binance_price(asset_info['binance'])
current_price = realtime_price if realtime_price else float(df['close'].iloc[-1])
current_atr = float(df['ATR'].iloc[-1])
is_mtf_bullish = bool(df['mtf_trend_bullish'].iloc[-1])

brussels_tz = ZoneInfo('Europe/Brussels')
laatste_update = datetime.now(brussels_tz).strftime('%H:%M:%S (%d-%m-%Y)')

st.subheader(f"{asset_info['icon']} Geselecteerd: {selected_asset_name} | Tijdschaal: {selected_tf_name}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('Huidige Koers (Live)', f'${current_price:,.3f}')
col2.metric('Model Score ($R^2$)', f'{score:.2f}')
col3.metric('Directional Win-Rate', f'{directional_acc:.1f}%')
col4.metric(
    'Daily Trend Filter', '🟢 BULLISH' if is_mtf_bullish else '🔴 BEARISH'
)
col5.metric('Laatste Call (Brussel)', laatste_update)

st.markdown('---')

latest_data = X.iloc[-1].values.reshape(1, -1)
predicted_price = model.predict(latest_data)[0]
price_diff = predicted_price - current_price
percentage_diff = (price_diff / current_price) * 100

stop_loss = (
    current_price - (1.5 * current_atr)
    if price_diff > 0
    else current_price + (1.5 * current_atr)
)
take_profit = (
    current_price + (2.5 * current_atr)
    if price_diff > 0
    else current_price - (2.5 * current_atr)
)

st.subheader(f'🔮 Voorspelling voor de volgende {selected_tf_name} Candle & Risk Management')

col_a, col_b, col_c = st.columns(3)
col_a.metric(
    f'Voorspelde Prijs ({selected_tf_name})',
    f'${predicted_price:,.3f}',
    delta=f'{percentage_diff:+.3f}%',
)
col_b.metric('Aanbevolen Stop-Loss', f'${stop_loss:,.3f}')
col_c.metric('Aanbevolen Take-Profit', f'${take_profit:,.3f}')

if predicted_price > current_price and is_mtf_bullish:
  st.success(
      f'**STERK LONG SIGNAAL 🚀** — Het model verwacht een stijging voor de volgende'
      f' {selected_tf_name} candle van {selected_asset_name} (dagtrend is positief).'
  )
elif predicted_price > current_price and not is_mtf_bullish:
  st.warning(
      f'**LET OP: Tegen-trend signaal ⚠️** — Model voorspelt stijging voor'
      f' {selected_asset_name}, maar de dagelijkse trend is negatief.'
  )
else:
  st.error(
      f'**SHORT SIGNAAL 📉** — Het model verwacht een daling over de komende'
      f' {selected_tf_name} candle voor {selected_asset_name}.'
  )

if send_alert and webhook_url:
  try:
    payload = {
        'content': (
            f"{asset_info['icon']} **{selected_asset_name} ({selected_tf_name}) Alert**"
            f' \nKoers: ${current_price:,.3f}\nDoel: ${predicted_price:,.3f}'
            f' ({percentage_diff:+.3f}%)\nSL: ${stop_loss:,.3f} | TP:'
            f' ${take_profit:,.3f}'
        )
    }
    requests.post(webhook_url, json=payload, timeout=5)
    st.sidebar.success('Signaal succesvol verzonden via webhook!')
  except Exception as e:
    st.sidebar.error(f'Fout bij versturen: {e}')

st.markdown('---')

# --- SHAP EXPLAINABILITY (Veilig ingepakt) ---
st.subheader('🔍 SHAP Model Explainability (Waarom deze voorspelling?)')
try:
  import shap
  explainer = shap.TreeExplainer(model)
  shap_values = explainer(X.tail(1))

  fig, ax = plt.subplots(figsize=(10, 4))
  shap.plots.waterfall(shap_values[0], max_display=8, show=False)
  st.pyplot(fig)
  plt.close(fig)
except Exception as e:
  st.info('SHAP-uitleg kon tijdelijk niet worden geladen (geen impact op model).')

st.markdown('---')

st.subheader('📊 Belangrijkste Features volgens XGBoost')
importances = pd.Series(model.feature_importances_, index=features).sort_values(
    ascending=True
)
st.bar_chart(importances)
