import json
import os
import time
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
import yfinance as yf

# Maak een gedeelde sessie aan met een browser User-Agent om rate-limits te vermijden
session = requests.Session()
session.headers['User-Agent'] = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like'
    ' Gecko) Chrome/120.0.0.0 Safari/537.36'
)

# 1. Definieer de assets en hun instellingen
assets = {
    'BTC': {
        'yf': 'BTC-USD',
        'state_file': 'model_state_btc.json',
        'model_file': 'model_btc.json',
    },
    'ETH': {
        'yf': 'ETH-USD',
        'state_file': 'model_state_eth.json',
        'model_file': 'model_eth.json',
    },
    'LINK': {
        'yf': 'LINK-USD',
        'state_file': 'model_state_link.json',
        'model_file': 'model_link.json',
    },
    'AAVE': {
        'yf': 'AAVE-USD',
        'state_file': 'model_state_aave.json',
        'model_file': 'model_aave.json',
    },
}


def load_or_init_state(state_file):
  """Laadt de bestaande state van de asset of initialiseert een nieuwe."""
  if os.path.exists(state_file):
    with open(state_file, 'r') as f:
      return json.load(f)
  else:
    return {
        'learning_rate': 0.05,
        'max_depth': 5,
        'n_estimators': 100,
        'previous_learning_rate': 0.05,
        'previous_max_depth': 5,
        'baseline_error': float('inf'),
        'last_prediction': None,
    }


def evaluate_and_adapt(state, current_price):
  """Evalueert de voorspelling van de vorige run en stuurt parameters aan."""
  if state['last_prediction'] is not None:
    error = abs(state['last_prediction'] - current_price)
    print(
        f"Vorige voorspelling: {state['last_prediction']:.2f} | Echte koers:"
        f' {current_price:.2f} | Absolute Fout: {error:.2f}'
    )

    if error < state['baseline_error']:
      print(
          '🚀 Mutatie was succesvol! Prestatie verbeterd, parameters behouden.'
      )
      state['baseline_error'] = error
    else:
      print(
          '⚠️ Mutatie maakte het slechter. Automatische Rollback naar veilige'
          ' parameters!'
      )
      state['learning_rate'] = state['previous_learning_rate']
      state['max_depth'] = state['previous_max_depth']

  state['previous_learning_rate'] = state['learning_rate']
  state['previous_max_depth'] = state['max_depth']

  mutation = np.random.choice([-0.005, 0.005], p=[0.5, 0.5])
  state['learning_rate'] = float(
      np.clip(state['learning_rate'] + mutation, 0.01, 0.1)
  )

  return state


# Features lijst
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
    'RVOL',
    'upper_wick',
    'lower_wick',
    'RSI_momentum',
    'MACD_hist_momentum',
]


def main():
  print('Start quant_worker.py...')
  for asset_name, info in assets.items():
    print(f'\n--- Verwerken van {asset_name} ---')
    try:
      # Haal data op met de aangepaste sessie om rate-limits te voorkomen
      df = yf.download(
          info['yf'], period='14d', interval='15m', progress=False, session=session
      )
      if df is None or df.empty:
        continue

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

      # Technische indicatoren en micro-features
      df['daily_close'] = df['close']
      df['daily_SMA50'] = df['close'].rolling(window=50).mean().bfill()

      sma_20 = df['close'].rolling(window=20).mean()
      std_20 = df['close'].rolling(window=20).std()
      upper_band = sma_20 + (std_20 * 2)
      lower_band = sma_20 - (std_20 * 2)
      df['BB_pct_b'] = (df['close'] - lower_band) / (
          upper_band - lower_band + 1e-8
      )

      high_low = df['high'] - df['low']
      high_close = np.abs(df['high'] - df['close'].shift())
      low_close = np.abs(df['low'] - df['close'].shift())
      true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
          axis=1
      )
      df['ATR'] = true_range.rolling(window=14).mean()

      df['SMA_20'] = sma_20
      df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
      df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
      df['MACD'] = df['EMA_9'] - df['EMA_21']
      df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

      delta = df['close'].diff()
      gain = delta.where(delta > 0, 0.0)
      loss = -delta.where(delta < 0, 0.0)
      rs = (
          gain.rolling(window=14).mean() / loss.rolling(window=14).mean()
      )
      df['RSI'] = 100 - (100 / (1 + rs))

      low_14 = df['low'].rolling(window=14).min()
      high_14 = df['high'].rolling(window=14).max()
      df['Stoch_K'] = (
          (df['close'] - low_14) / (high_14 - low_14 + 1e-8)
      ) * 100
      df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

      df['OBV'] = (
          (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
      )

      up_move = df['high'] - df['high'].shift()
      down_move = df['low'].shift() - df['low']
      plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
      minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
      plus_dm_s = (
          pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean()
      )
      minus_dm_s = (
          pd.Series(minus_dm, index=df.index)
          .ewm(alpha=1 / 14, adjust=False)
          .mean()
      )
      tr_s = true_range.ewm(alpha=1 / 14, adjust=False).mean()
      plus_di = 100 * (plus_dm_s / (tr_s + 1e-8))
      minus_di = 100 * (minus_dm_s / (tr_s + 1e-8))
      dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
      df['ADX'] = dx.ewm(alpha=1 / 14, adjust=False).mean()

      rolling_high = df['high'].rolling(window=50).max()
      rolling_low = df['low'].rolling(window=50).min()
      diff_hl = rolling_high - rolling_low
      df['Fib_382'] = rolling_high - (diff_hl * 0.382)
      df['Fib_500'] = rolling_high - (diff_hl * 0.500)
      df['Fib_618'] = rolling_high - (diff_hl * 0.618)
      df['Fib_618_distance'] = (df['close'] - df['Fib_618']) / (
          df['Fib_618'] + 1e-8
      )

      typical_price = (df['high'] + df['low'] + df['close']) / 3
      df['VWAP'] = (typical_price * df['volume']).rolling(
          window=96 * 2
      ).sum() / (df['volume'].rolling(window=96 * 2).sum() + 1e-8)
      df['VWAP_distance'] = (df['close'] - df['VWAP']) / (df['VWAP'] + 1e-8)

      df['mtf_trend_bullish'] = (df['daily_close'] > df['daily_SMA50']).astype(
          int
      )
      df['fear_greed'] = 50.0

      df['RVOL'] = df['volume'] / (
          df['volume'].rolling(window=20).mean() + 1e-8
      )
      body_size = (df['close'] - df['open']).abs() + 1e-8
      df['upper_wick'] = (
          df['high'] - df[['open', 'close']].max(axis=1)
      ) / body_size
      df['lower_wick'] = (
          df[['open', 'close']].min(axis=1) - df['low']
      ) / body_size
      df['RSI_momentum'] = df['RSI'].diff()
      macd_histogram = df['MACD'] - df['MACD_Signal']
      df['MACD_hist_momentum'] = macd_histogram.diff()

      df_clean = df.dropna(subset=features + ['close']).reset_index(drop=True)
      if df_clean.empty:
        continue

      current_price = float(df_clean['close'].iloc[-1])

      state = load_or_init_state(info['state_file'])
      state = evaluate_and_adapt(state, current_price)

      X = df_clean[features].iloc[:-1]
      y = df_clean['close'].shift(-1).dropna()
      X = X.iloc[: len(y)]

      model = xgb.XGBRegressor(
          learning_rate=state['learning_rate'],
          max_depth=state['max_depth'],
          n_estimators=state['n_estimators'],
          random_state=42,
      )
      model.fit(X, y)

      latest_features = df_clean[features].iloc[-1].values.reshape(1, -1)
      next_prediction = float(model.predict(latest_features)[0])
      state['last_prediction'] = next_prediction

      model.save_model(info['model_file'])

      with open(info['state_file'], 'w') as f:
        json.dump(state, f, indent=4)

      print(
          f'[{asset_name}] Nieuwe voorspelling opgeslagen:'
          f" {next_prediction:.2f} (LR: {state['learning_rate']:.4f})"
      )

      # Wacht 3 seconden per asset om rate limits (HTTP 429) te voorkomen
      time.sleep(3)

    except Exception as e:
      print(f'Fout bij verwerken van {asset_name}: {e}')

  print('Worker succesvol gedraaid!')


if __name__ == '__main__':
  main()
