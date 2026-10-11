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

st.title('⚡ Institutional Multi-Asset Multi-Timeframe Quant Model')
st.markdown(
    'Professioneel quant dashboard met **Multi-Asset Overzichtstabel**, **Confluence Score**, '
    '**Sharpe/Sortino Backtesting**, **ATR Risk Management** en **SHAP Explainability** — 100% feitelijke data.'
)

# --- ASSET & TIJDSCHAAL CONFIGURATIE ---
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

timeframe_options = {
    '15 Minuten (15m)': {'interval': '15m', 'period': '30d', 'bars_per_day': 96},
    '30 Minuten (30m)': {'interval': '30m', 'period': '60d', 'bars_per_day': 48},
    '1 Uur (1h)': {'interval': '1h', 'period': '730d', 'bars_per_day': 24},
    '4 Uur (4h)': {'interval': '1h', 'period': '730d', 'bars_per_day': 6},
    '1 Dag (1d)': {'interval': '1d', 'period': 'max', 'bars_per_day': 1},
    '1 Week (1wk)': {'interval': '1wk', 'period': 'max', 'bars_per_day': 1},
    '1 Maand (1mo)': {'interval': '1mo', 'period': 'max', 'bars_per_day': 1},
}

st.sidebar.header('⚙️ Dashboard Instellingen')
selected_asset_name = st.sidebar.selectbox('Selecteer Hoofdasset', list(asset_options.keys()))
asset_info = asset_options[selected_asset_name]

selected_tf_name = st.sidebar.selectbox('Selecteer Tijdschaal (Candle)', list(timeframe_options.keys()))
tf_info = timeframe_options[selected_tf_name]

test_size = st.sidebar.slider('Test / Train Splitsing (%)', 10, 40, 20, step=5)
n_estimators = st.sidebar.slider('XGBoost Bomen (n_estimators)', 50, 300, 150)
max_depth = st.sidebar.slider('Maximale Boomdiepte (max_depth)', 3, 10, 4)

st.sidebar.markdown('---')
st.sidebar.subheader('🤖 Notificatie / Webhook (Optioneel)')
webhook_url = st.sidebar.text_input('Discord/Telegram Webhook URL')
send_alert = st.sidebar.button('Stuur Signaal Test')


# --- DATA FUNCTIES ---
@st.cache_data(ttl=300)
def load_market_data(ticker, interval, period):
  try:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if (df is None or df.empty) and ticker == 'CC-USD':
      df = yf.download('CC1-USD', period=period, interval=interval, progress=False)
    if df is None or df.empty:
      return None

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={date_col: 'Date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').dropna(subset=['close']).reset_index(drop=True)

    df_daily = yf.download(ticker, period='max', interval='1d', progress=False)
    if (df_daily is None or df_daily.empty) and ticker == 'CC-USD':
      df_daily = yf.download('CC1-USD', period='max', interval='1d', progress=False)

    if df_daily is not None and not df_daily.empty:
      if isinstance(df_daily.columns, pd.MultiIndex):
        df_daily.columns = df_daily.columns.get_level_values(0)
      df_daily = df_daily.reset_index()
      d_col = 'Datetime' if 'Datetime' in df_daily.columns else 'Date'
      df_daily = df_daily.rename(columns={d_col: 'Date', 'Close': 'daily_close'})
      df_daily['Date'] = pd.to_datetime(df_daily['Date'])
      df_daily['daily_SMA50'] = df_daily['daily_close'].rolling(window=50).mean()

      df['Date_only'] = df['Date'].dt.date
      df_daily['Date_only'] = df_daily['Date'].dt.date
      df = pd.merge(df, df_daily[['Date_only', 'daily_close', 'daily_SMA50']], on='Date_only', how='left')
      df = df.drop(columns=['Date_only']).ffill()

    if 'daily_close' not in df.columns:
      df['daily_close'] = df['close']
      df['daily_SMA50'] = df['close']

    df = df.dropna().reset_index(drop=True)
    return df if not df.empty else None
  except Exception:
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


# ==========================================
# DEEL 1: MULTI-ASSET OVERZICHTSTABEL (GRID)
# ==========================================
st.subheader('🌐 Multi-Asset Real-Time Overzicht & Signalen')
overview_data = []

with st.spinner('Bezig met doorlichten van alle assets in de suite...'):
  fng_val = fetch_crypto_fear_greed()
  for name, info in asset_options.items():
    df_temp = load_market_data(info['yf'], tf_info['interval'], tf_info['period'])
    if df_temp is not None and len(df_temp) > 30:
      df_temp = add_advanced_indicators(df_temp, tf_info['bars_per_day'])
      df_temp['fear_greed'] = fng_val
      df_temp['Target'] = df_temp['close'].shift(-1)
      df_temp = df_temp.dropna().reset_index(drop=True)
      
      X_t = df_temp[features]
      y_t = df_temp['Target']
      X_tr, X_te, y_tr, y_te = train_test_split(X_t, y_t, test_size=0.2, shuffle=False)
      
      m_temp = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
      m_temp.fit(X_tr, y_tr)
      
      live_p = fetch_realtime_binance_price(info['binance'])
      curr_p = live_p if live_p else float(df_temp['close'].iloc[-1])
      pred_p = m_temp.predict(X_t.tail(1))[0]
      pct_d = ((pred_p - curr_p) / curr_p) * 100
      is_bull = bool(df_temp['mtf_trend_bullish'].iloc[-1])
      
      t_preds = m_temp.predict(X_te)
      dir_acc = np.mean(np.sign(t_preds - X_te['close'].values) == np.sign(y_te.values - X_te['close'].values)) * 100
      
      signal = '🚀 STERK LONG' if pct_d > 0 and is_bull else ('⚠️ CONTRA-LONG' if pct_d > 0 else '📉 SHORT')
      
      overview_data.append({
          'Asset': f"{info['icon']} {name}",
          'Koers': f"${curr_p:,.3f}",
          'Voorspelling': f"${pred_p:,.3f} ({pct_d:+.2f}%)",
          'Signaal': signal,
          'Dagtrend': '🟢 Bullish' if is_bull else '🔴 Bearish',
          'Win-Rate': f"{dir_acc:.1f}%"
      })
    else:
      overview_data.append({
          'Asset': f"{info['icon']} {name}",
          'Koers': 'N.B. / Geen Data',
          'Voorspelling': '-',
          'Signaal': 'Geen data beschikbaar',
          'Dagtrend': '-',
          'Win-Rate': '-'
      })

st.dataframe(pd.DataFrame(overview_data), use_container_width=True)
st.markdown('---')


# ==========================================
# DEEL 2: DIEPE ANALYSE VOOR GEKOZEN ASSET
# ==========================================
with st.spinner(f'Laden van gedetailleerde quant data voor {selected_asset_name}...'):
  df = load_market_data(asset_info['yf'], tf_info['interval'], tf_info['period'])
  if df is None or df.empty or len(df) < 30:
    st.error(f"Onvoldoende historische data voor {selected_asset_name} op tijdschaal {selected_tf_name}.")
    st.stop()

  df = add_advanced_indicators(df, tf_info['bars_per_day'])
  df['fear_greed'] = fng_val
  df['Target'] = df['close'].shift(-1)
  df = df.dropna().reset_index(drop=True)

X = df[features]
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100.0, shuffle=False)

model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

test_preds = model.predict(X_test)
test_actuals = y_test.values
test_closes = X_test['close'].values
pred_direction = np.sign(test_preds - test_closes)
actual_direction = np.sign(test_actuals - test_closes)
directional_acc = np.mean(pred_direction == actual_direction) * 100

last_rsi = float(df['RSI'].iloc[-1])
last_macd = float(df['MACD'].iloc[-1])
is_mtf_bullish = bool(df['mtf_trend_bullish'].iloc[-1])
confluence_points = 0
if pred_direction[-1] > 0 and is_mtf_bullish: confluence_points += 40
if 40 <= last_rsi <= 65: confluence_points += 30
if last_macd > 0: confluence_points += 30
confluence_score = min(100, max(10, confluence_points if pred_direction[-1] > 0 else 100 - confluence_points))

realtime_price = fetch_realtime_binance_price(asset_info['binance'])
current_price = realtime_price if realtime_price else float(df['close'].iloc[-1])
current_atr = float(df['ATR'].iloc[-1])

brussels_tz = ZoneInfo('Europe/Brussels')
laatste_update = datetime.now(brussels_tz).strftime('%H:%M:%S (%d-%m-%Y)')

st.subheader(f"{asset_info['icon']} Geselecteerde Hoofdasset: {selected_asset_name} ({selected_tf_name})")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('Huidige Koers', f'${current_price:,.3f}')
col2.metric('Model Score ($R^2$)', f'{score:.2f}')
col3.metric('Directional Win-Rate', f'{directional_acc:.1f}%')
col4.metric('Confluence Score', f'{confluence_score}%')
col5.metric('Laatste Call (Brussel)', laatste_update)

st.markdown('---')

latest_data = X.iloc[-1].values.reshape(1, -1)
predicted_price = model.predict(latest_data)[0]
price_diff = predicted_price - current_price
percentage_diff = (price_diff / current_price) * 100

stop_loss = current_price - (1.5 * current_atr) if price_diff > 0 else current_price + (1.5 * current_atr)
take_profit = current_price + (2.5 * current_atr) if price_diff > 0 else current_price - (2.5 * current_atr)

st.subheader(f'🔮 Voorspelling & Risk Management ({selected_tf_name} Candle)')

col_a, col_b, col_c = st.columns(3)
col_a.metric(f'Voorspelde Prijs', f'${predicted_price:,.3f}', delta=f'{percentage_diff:+.3f}%')
col_b.metric('Aanbevolen Stop-Loss', f'${stop_loss:,.3f}')
col_c.metric('Aanbevolen Take-Profit', f'${take_profit:,.3f}')

if predicted_price > current_price and is_mtf_bullish:
  st.success(f'**STERK LONG SETUP 🚀** — Model verwacht stijging en Confluence is hoog ({confluence_score}%).')
elif predicted_price > current_price and not is_mtf_bullish:
  st.warning(f'**CONTRA-TREND LONG ⚠️** — Model voorspelt stijging, maar dagtrend is bearish.')
else:
  st.error(f'**SHORT SETUP 📉** — Model verwacht daling voor de komende candle.')

if send_alert and webhook_url:
  try:
    payload = {'content': f"{asset_info['icon']} **{selected_asset_name} Alert** \nKoers: ${current_price:,.3f}\nDoel: ${predicted_price:,.3f} ({percentage_diff:+.3f}%)"}
    requests.post(webhook_url, json=payload, timeout=5)
    st.sidebar.success('Signaal verzonden!')
  except Exception as e:
    st.sidebar.error(f'Fout: {e}')

st.markdown('---')


# ==========================================
# DEEL 3: BACKTEST PERFORMANCE & RATIOS
# ==========================================
st.subheader('📈 Geavanceerde Backtest prestaties & Ratios (Testset)')

strategy_returns = np.where(pred_direction > 0, (test_actuals - test_closes) / test_closes, - (test_actuals - test_closes) / test_closes)
strat_cum = np.cumprod(1 + strategy_returns) - 1
bh_cum = np.cumprod(1 + (test_actuals - test_closes) / test_closes) - 1

daily_strat_ret = pd.Series(strategy_returns)
sharpe_ratio = (daily_strat_ret.mean() / (daily_strat_ret.std() + 1e-8)) * np.sqrt(252)
downside_std = daily_strat_ret[daily_strat_ret < 0].std()
sortino_ratio = (daily_strat_ret.mean() / (downside_std + 1e-8)) * np.sqrt(252)
max_drawdown = (np.maximum.accumulate(1 + strat_cum) - (1 + strat_cum)).max() * 100

col_r1, col_r2, col_r3 = st.columns(3)
col_r1.metric('Sharpe Ratio (Risico-rendement)', f'{sharpe_ratio:.2f}')
col_r2.metric('Sortino Ratio (Downside)', f'{sortino_ratio:.2f}')
col_r3.metric('Maximum Drawdown', f'-{max_drawdown:.2f}%')

fig_bt, ax_bt = plt.subplots(figsize=(10, 3))
ax_bt.plot(strat_cum * 100, label='Quant Strategie Rendement (%)', color='#00CC66', lw=2)
ax_bt.plot(bh_cum * 100, label='Buy & Hold Benchmark (%)', color='#888888', ls='--', lw=1.5)
ax_bt.set_ylabel('Rendement (%)')
ax_bt.legend(loc='upper left')
ax_bt.grid(True, alpha=0.3)
st.pyplot(fig_bt)
plt.close(fig_bt)

st.markdown('---')

# --- SHAP & FEATURE IMPORTANCE ---
col_shap, col_feat = st.columns(2)

with col_shap:
  st.subheader('🔍 SHAP Explainability')
  try:
    import shap
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X.tail(1))
    fig_s, ax_s = plt.subplots(figsize=(6, 3))
    shap.plots.waterfall(shap_vals[0], max_display=6, show=False)
    st.pyplot(fig_s)
    plt.close(fig_s)
  except Exception:
    st.info('SHAP-uitleg tijdelijk niet beschikbaar.')

with col_feat:
  st.subheader('📊 Belangrijkste Features')
  importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
  fig_f, ax_f = plt.subplots(figsize=(6, 3))
  importances.tail(6).plot(kind='barh', ax=ax_f, color='#1f77b4')
  ax_f.grid(True, alpha=0.3)
  st.pyplot(fig_f)
  plt.close(fig_f)
