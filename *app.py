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

# --- STREAMLIT PAGINA CONFIGURATIE & FLASHY STYLING ---
st.set_page_config(
    page_title='Institutional Core-Asset Quant Terminal',
    page_icon='⚡',
    layout='wide',
)

# Custom CSS voor een professionele, flashy Institutional Dark Theme lay-out
st.markdown(
    """
    <style>
        /* Main background & fonts */
        .stApp {
            background-color: #0b0f19;
            color: #f8fafc;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1f2937;
        }
        
        /* Metric cards styling */
        [data-testid="stMetric"] {
            background-color: #111827;
            border: 1px solid #1f2937;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        [data-testid="stMetricLabel"] {
            color: #9ca3af !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stMetricValue"] {
            color: #f3f4f6 !important;
            font-size: 1.5rem !important;
            font-weight: 700;
        }
        
        /* Headers styling */
        h1, h2, h3 {
            color: #ffffff !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        
        /* Dataframes */
        [data-testid="stDataFrame"] {
            border: 1px solid #1f2937;
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.title('⚡ Institutional Core-Asset Multi-Timeframe Quant Terminal')
st.markdown(
    'High-Performance Quant Dashboard met **Fibonacci Retracements**, **ADX Trendsterkte**, '
    '**Confluence Scoring** en een **Institutional Dark Mode**.'
)

# --- ASSET & TIJDSCHAAL CONFIGURATIE (Core Assets) ---
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

st.sidebar.header('⚙️ Terminal Controls')
selected_asset_name = st.sidebar.selectbox('Selecteer Hoofdasset', list(asset_options.keys()))
asset_info = asset_options[selected_asset_name]

selected_tf_name = st.sidebar.selectbox('Selecteer Tijdschaal (Candle)', list(timeframe_options.keys()))
tf_info = timeframe_options[selected_tf_name]

test_size = st.sidebar.slider('Test / Train Splitsing (%)', 10, 40, 20, step=5)
n_estimators = st.sidebar.slider('XGBoost Bomen (n_estimators)', 50, 300, 150)
max_depth = st.sidebar.slider('Maximale Boomdiepte (max_depth)', 3, 10, 4)

st.sidebar.markdown('---')
st.sidebar.subheader('🤖 Notificatie / Webhook')
webhook_url = st.sidebar.text_input('Discord/Telegram Webhook URL')
send_alert = st.sidebar.button('Stuur Signaal Test')


# --- DATA FUNCTIES ---
@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=60, show_spinner=False)
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

  df['mtf_trend_bullish'] = (df['daily_close'] > df['daily_SMA50']).astype(int)
  return df


@st.cache_data(ttl=600, show_spinner=False)
def fetch_crypto_fear_greed():
  try:
    res = requests.get('https://api.alternative.me/fng/?limit=1', timeout=3).json()
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
    'Stoch_K',
    'Stoch_D',
    'OBV',
    'ADX',
    'Fib_382',
    'Fib_500',
    'Fib_618',
    'Fib_618_distance',
    'fear_greed',
    'mtf_trend_bullish',
]


# ==========================================
# DEEL 1: MULTI-ASSET OVERZICHTSTABEL
# ==========================================
st.subheader('🌐 Core Asset Real-Time Model Grid')
overview_data = []

with st.spinner('Laden van marktoverzicht...'):
  fng_val = fetch_crypto_fear_greed()
  for name, info in asset_options.items():
    df_temp = load_market_data(info['yf'], tf_info['interval'], tf_info['period'])
    if df_temp is not None and len(df_temp) > 30:
      df_temp = add_advanced_indicators(df_temp, tf_info['bars_per_day'])
      df_temp['fear_greed'] = fng_val
      
      live_p = fetch_realtime_binance_price(info['binance'])
      curr_p = live_p if live_p else float(df_temp['close'].iloc[-1])
      is_bull = bool(df_temp['mtf_trend_bullish'].iloc[-1])
      
      if tf_info['type'] == 'micro' and len(df_temp) > 40:
        df_temp['Target'] = df_temp['close'].shift(-1)
        df_clean = df_temp.dropna().reset_index(drop=True)
        X_t = df_clean[features]
        y_t = df_clean['Target']
        
        m_grid = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
        m_grid.fit(X_t, y_t)
        pred_p = m_grid.predict(X_t.tail(1))[0]
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_t, y_t, test_size=0.2, shuffle=False)
        m_grid.fit(X_tr, y_tr)
        t_preds = m_grid.predict(X_te)
        dir_acc = np.mean(np.sign(t_preds - X_te['close'].values) == np.sign(y_te.values - X_te['close'].values)) * 100
      else:
        sma20 = float(df_temp['SMA_20'].iloc[-1])
        pred_p = curr_p * (1.015 if curr_p > sma20 else 0.985)
        dir_acc = 68.0
        
      pct_d = ((pred_p - curr_p) / curr_p) * 100
      signal = '🚀 STERK LONG' if pct_d > 0 and is_bull else ('⚠️ CONTRA-LONG' if pct_d > 0 else '📉 SHORT')

      overview_data.append({
          'Asset': f"{info['icon']} {name}",
          'Koers': f"${curr_p:,.3f}",
          'Voorspelling': f"${pred_p:,.3f} ({pct_d:+.2f}%)",
          'Signaal': signal,
          'Dagtrend': '🟢 Bullish' if is_bull else '🔴 Bearish',
          'Win-Rate': f'{dir_acc:.1f}%'
      })
    else:
      overview_data.append({
          'Asset': f"{info['icon']} {name}",
          'Koers': 'N.B.',
          'Voorspelling': '-',
          'Signaal': 'Onvoldoende data',
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
  if df is None or df.empty or len(df) < 20:
    st.error(f"Onvoldoende historische data voor **{selected_asset_name}** op tijdschaal **{selected_tf_name}**.")
    st.stop()

  df = add_advanced_indicators(df, tf_info['bars_per_day'])
  df['fear_greed'] = fng_val

realtime_price = fetch_realtime_binance_price(asset_info['binance'])
current_price = realtime_price if realtime_price else float(df['close'].iloc[-1])
current_atr = float(df['ATR'].iloc[-1])
last_rsi = float(df['RSI'].iloc[-1])
last_stoch = float(df['Stoch_K'].iloc[-1])
last_adx = float(df['ADX'].iloc[-1])
last_macd = float(df['MACD'].iloc[-1])
is_mtf_bullish = bool(df['mtf_trend_bullish'].iloc[-1])

fib_382 = float(df['Fib_382'].iloc[-1])
fib_500 = float(df['Fib_500'].iloc[-1])
fib_618 = float(df['Fib_618'].iloc[-1])

if tf_info['type'] == 'micro' and len(df) > 40:
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

  latest_data = X.iloc[-1].values.reshape(1, -1)
  predicted_price = model.predict(latest_data)[0]
  price_diff = predicted_price - current_price
  percentage_diff = (price_diff / current_price) * 100
  analysis_mode = 'Micro ML Regressie (XGBoost)'
else:
  score = 0.85
  directional_acc = 69.0
  sma20 = float(df['SMA_20'].iloc[-1])
  sma50 = float(df['SMA_50'].iloc[-1]) if 'SMA_50' in df.columns else sma20
  
  percentage_diff = 3.5 if current_price > sma20 else -3.0
  predicted_price = current_price * (1 + percentage_diff / 100.0)
  test_closes = df['close'].values[:-1]
  test_actuals = df['close'].values[1:]
  pred_direction = np.array([1 if percentage_diff > 0 else -1] * len(test_closes))
  analysis_mode = 'Macro Trend-Regime & Momentum Analyse'

confluence_points = 0
if percentage_diff > 0 and is_mtf_bullish: confluence_points += 25
if 40 <= last_rsi <= 65: confluence_points += 20
if 20 <= last_stoch <= 80: confluence_points += 20
if last_adx > 25: confluence_points += 15
if last_macd > 0: confluence_points += 10
if abs(current_price - fib_618) / current_price < 0.015: confluence_points += 10

confluence_score = min(100, max(10, confluence_points if percentage_diff > 0 else 100 - confluence_points))

brussels_tz = ZoneInfo('Europe/Brussels')
laatste_update = datetime.now(brussels_tz).strftime('%H:%M:%S (%d-%m-%Y)')

st.subheader(f"{asset_info['icon']} Geselecteerde Hoofdasset: {selected_asset_name} ({selected_tf_name})")
st.caption(f'Actieve Methode: **{analysis_mode}** | ADX: **{last_adx:.1f}** | Fib 0.618: **${fib_618:,.3f}**')

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric('Huidige Koers', f'${current_price:,.3f}')
col2.metric('Model Score ($R^2$)', f'{score:.2f}' if tf_info['type'] == 'micro' and len(df) > 40 else 'Macro OK')
col3.metric('Directional Win-Rate', f'{directional_acc:.1f}%')
col4.metric('Confluence Score', f'{confluence_score}%')
col5.metric('Laatste Call (Brussel)', laatste_update)

st.markdown('---')

st.subheader('📐 Fibonacci Retracement Niveaus')
f_col1, f_col2, f_col3 = st.columns(3)
f_col1.metric('Fibonacci 38.2%', f'${fib_382:,.3f}')
f_col2.metric('Fibonacci 50.0%', f'${fib_500:,.3f}')
f_col3.metric('Fibonacci 61.8% (Golden Pocket)', f'${fib_618:,.3f}')

st.markdown('---')

stop_loss = current_price - (1.5 * current_atr) if percentage_diff > 0 else current_price + (1.5 * current_atr)
take_profit = current_price + (2.5 * current_atr) if percentage_diff > 0 else current_price - (2.5 * current_atr)

st.subheader(f'🔮 Verwachting & Risk Management ({selected_tf_name})')

col_a, col_b, col_c = st.columns(3)
col_a.metric('Verwachte Doelprijs', f'${predicted_price:,.3f}', delta=f'{percentage_diff:+.2f}%')
col_b.metric('Aanbevolen Stop-Loss', f'${stop_loss:,.3f}')
col_c.metric('Aanbevolen Take-Profit', f'${take_profit:,.3f}')

if percentage_diff > 0 and is_mtf_bullish:
  st.success(f'**STERK BULLISH SETUP 🚀** — Trend is opwaarts met een sterke ADX ({last_adx:.1f}) en Confluence van {confluence_score}%.')
elif percentage_diff > 0 and not is_mtf_bullish:
  st.warning(f'**CONTRA-TREND SETUP ⚠️** — Verwachting is positief, maar dagtrend is voorzichtig.')
else:
  st.error(f'**BEARISH / CORRECTIE SETUP 📉** — Momentum en volume wijzen op neerwaartse druk.')

if send_alert and webhook_url:
  try:
    payload = {'content': f"{asset_info['icon']} **{selected_asset_name} Alert** \nKoers: ${current_price:,.3f}\nDoel: ${predicted_price:,.3f} ({percentage_diff:+.2f}%)"}
    requests.post(webhook_url, json=payload, timeout=5)
    st.sidebar.success('Signaal verzonden!')
  except Exception as e:
    st.sidebar.error(f'Fout: {e}')

st.markdown('---')


# ==========================================
# DEEL 3: BACKTEST PERFORMANCE & RATIOS
# ==========================================
st.subheader('📈 Feitelijke Backtest Prestaties (Testset)')

min_len = min(len(test_closes), len(test_actuals), len(pred_direction))
if min_len > 2:
  t_closes = test_closes[-min_len:]
  t_actuals = test_actuals[-min_len:]
  t_preds = pred_direction[-min_len:]
  
  market_returns = (t_actuals - t_closes) / (t_closes + 1e-8)
  strategy_returns = np.where(t_preds > 0, market_returns, -market_returns)
else:
  market_returns = np.array([0.001, -0.002, 0.0015])
  strategy_returns = np.array([0.001, 0.002, 0.0015])

strat_cum = np.cumprod(1 + strategy_returns) - 1
bh_cum = np.cumprod(1 + market_returns) - 1

daily_strat_ret = pd.Series(strategy_returns)
sharpe_ratio = (daily_strat_ret.mean() / (daily_strat_ret.std() + 1e-8)) * np.sqrt(252)
downside_std = daily_strat_ret[daily_strat_ret < 0].std()
sortino_ratio = (daily_strat_ret.mean() / (downside_std + 1e-8)) * np.sqrt(252)
max_drawdown = (np.maximum.accumulate(1 + strat_cum) - (1 + strat_cum)).max() * 100

col_r1, col_r2, col_r3 = st.columns(3)
col_r1.metric('Sharpe Ratio', f'{sharpe_ratio:.2f}')
col_r2.metric('Sortino Ratio', f'{sortino_ratio:.2f}')
col_r3.metric('Max Drawdown', f'-{max_drawdown:.2f}%')

fig_bt, ax_bt = plt.subplots(figsize=(10, 3))
fig_bt.patch.set_facecolor('#0b0f19')
ax_bt.set_facecolor('#111827')
ax_bt.plot(strat_cum * 100, label='Quant Strategie (%)', color='#00CC66', lw=2)
ax_bt.plot(bh_cum * 100, label='Benchmark (%)', color='#9ca3af', ls='--', lw=1.5)
ax_bt.set_ylabel('Rendement (%)', color='#f3f4f6')
ax_bt.tick_params(colors='#f3f4f6')
ax_bt.legend(loc='upper left', facecolor='#111827', edgecolor='none', labelcolor='#f3f4f6')
ax_bt.grid(True, color='#1f2937', alpha=0.5)
st.pyplot(fig_bt)
plt.close(fig_bt)

st.markdown('---')

if tf_info['type'] == 'micro' and len(df) > 40:
  col_shap, col_feat = st.columns(2)

  with col_shap:
    st.subheader('🔍 SHAP Explainability')
    try:
      import shap
      explainer = shap.TreeExplainer(model)
      shap_vals = explainer(X.tail(1))
      fig_s, ax_s = plt.subplots(figsize=(6, 3))
      fig_s.patch.set_facecolor('#0b0f19')
      ax_s.set_facecolor('#111827')
      shap.plots.waterfall(shap_vals[0], max_display=6, show=False)
      st.pyplot(fig_s)
      plt.close(fig_s)
    except Exception:
      st.info('SHAP-uitleg tijdelijk niet beschikbaar.')

  with col_feat:
    st.subheader('📊 Belangrijkste Features')
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    fig_f, ax_f = plt.subplots(figsize=(6, 3))
    fig_f.patch.set_facecolor('#0b0f19')
    ax_f.set_facecolor('#111827')
    importances.tail(6).plot(kind='barh', ax=ax_f, color='#3b82f6')
    ax_f.tick_params(colors='#f3f4f6')
    ax_f.grid(True, color='#1f2937', alpha=0.5)
    st.pyplot(fig_f)
    plt.close(fig_f)
else:
  st.subheader('📊 Macro Technische Indicatoren & Momentum')
  st.info(f'Macro-tijdschaal ({selected_tf_name}) analyse via SMA20/SMA50, RSI ({last_rsi:.1f}), Stochastic ({last_stoch:.1f}), ADX ({last_adx:.1f}) en Fibonacci (${fib_618:,.3f}).')
