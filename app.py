import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Pagina configuratie
st.set_page_config(
    page_title="Crypto Quant Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ RV Multi-Timeframe Ensemble Predictor")
st.markdown("*Real-time ensemble voorspellingen (XGBoost + LightGBM + CatBoost) gevoed door yfinance data.*")

# Sidebar: Asset keuze & Tijdinterval (Timescales)
asset_options = {
    'Bitcoin (BTC)': 'BTC',
    'Ethereum (ETH)': 'ETH',
    'Chainlink (LINK)': 'LINK',
    'Aave (AAVE)': 'AAVE',
}

# 4 timescales zoals afgesproken
interval_options = {
    '15 Minuten': {'interval': '15m', 'period': '59d', 'code': '15m'},
    '1 Uur': {'interval': '1h', 'period': '730d', 'code': '1h'},
    '4 Uur': {'interval': '1h', 'period': '730d', 'resample': '4h', 'code': '4h'},
    '1 Dag': {'interval': '1d', 'period': 'max', 'code': '1d'}
}
}

selected_asset_name = st.sidebar.selectbox("Selecteer Asset", list(asset_options.keys()))
selected_interval_name = st.sidebar.selectbox("Selecteer Tijdinterval", list(interval_options.keys()))

asset_code = asset_options[selected_asset_name]
interval_info = interval_options[selected_interval_name]
interval_code = interval_info['code']

@st.cache_data(ttl=300)
def load_dashboard_data(ticker, tf_info):
    """Haalt de basisdata op en resamplet 1h data naar 4h candles indien gevraagd."""
    try:
        # tf_info['interval'] is hier '1h' voor de 4h optie
        df = yf.download(ticker, period=tf_info['period'], interval=tf_info['interval'], progress=False)
        if df is None or df.empty:
            return None
            
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
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Converteer 1h data naar 4h candles
        if tf_info.get('resample') == '4h':
            df = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            df.dropna(inplace=True)
        
        # Technische indicatoren
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        rs = gain.rolling(window=14).mean() / (loss.rolling(window=14).mean() + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df.dropna()
    except Exception:
        return None

# Haal live grafiekdata op
ticker = f"{asset_code}-USD"
df = load_dashboard_data(ticker, interval_info)

# State bestand naam volgt nu het formaat: model_state_{asset}_{timescale}.json
state_file = f"model_state_{asset_code.lower()}_{interval_code}.json"

if df is not None and not df.empty:
    current_price = float(df['close'].iloc[-1])
    
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            predicted_price = state.get('last_prediction')
            lr = state.get('learning_rate', 0.05)
            baseline_err = state.get('baseline_error', 0)
            
            if predicted_price is not None:
                pct_diff = ((predicted_price - current_price) / current_price) * 100
                
                # Weergave met metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Huidige Koers (yfinance)", f"${current_price:,.2f}")
                col2.metric(f"Ensemble Voorspelling ({interval_code})", f"${predicted_price:,.2f}", delta=f"{pct_diff:+.2f}%")
                col3.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
                col4.metric("Adaptieve LR", f"${lr:.4f}" if isinstance(lr, float) else f"{lr}")
            else:
                st.warning("Model state gevonden, maar nog geen voorspelling beschikbaar.")
                
            st.subheader(f"Prijsverloop ({interval_code}) - {selected_asset_name}")
            st.line_chart(df['close'])
            
            with st.expander("📄 Bekijk Model & State Details"):
                st.json(state)
                
        except Exception as e:
            st.error(f"Fout bij het inlezen van state-bestand: {e}")
    else:
        st.warning(f"Model state-bestand `{state_file}` nog niet gevonden. Wacht op de eerste run van de achtergrond-worker.")
else:
    st.error("Kon geen marktdata ophalen via yfinance.")
