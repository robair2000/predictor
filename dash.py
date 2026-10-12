import json
import os
import streamlit as st

# Pagina configuratie
st.set_page_config(
    page_title='RV Quant Dashboard', page_icon='📈', layout='wide'
)

st.title('🚀 RV Institutional Quant Dashboard')
st.markdown(
    'Real-time multi-timeframe overzicht van je zelflerende ensemble worker.'
)

# Assets en tijdsbestekken die we beheren
assets = ['BTC', 'ETH', 'LINK', 'AAVE']
timeframes = ['15m', '1h', '1d', '1w', '1m']

def load_state(asset_name, tf_name):
    state_file = f'model_state_{asset_name.lower()}_{tf_name}.json'
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

# Sidebar navigatie of info
st.sidebar.header('Besturing')
if st.sidebar.button('🔄 Gegevens verversen'):
    st.rerun()

st.sidebar.markdown('---')
st.sidebar.info(
    'Dit dashboard leest de geavanceerde JSON state-bestanden uit '
    'die door je Render worker via GitHub worden bijgewerkt.'
)

# Layout per asset in kolommen
cols = st.columns(len(assets))

for i, asset in enumerate(assets):
    with cols[i]:
        st.subheader(f'🪙 {asset}')
        
        # Maak tabs voor de verschillende tijdsbestekken per asset
        tf_tabs = st.tabs(timeframes)
        
        for j, tf in enumerate(timeframes):
            with tf_tabs[j]:
                state = load_state(asset, tf)
                
                if state:
                    last_pred = state.get('last_prediction')
                    lr = state.get('learning_rate')
                    prob = state.get('direction_probability', 50.0)
                    lookback = state.get('lookback_window', 150)
                    sl = state.get('stop_loss', 0.0)
                    tp = state.get('take_profit', 0.0)

                    # Prijsvoorspelling metric
                    if last_pred is not None:
                        st.metric(
                            label='Voorspelde Prijs', value=f'$ {last_pred:,.2f}'
                        )
                    else:
                        st.metric(label='Voorspelde Prijs', value='Nog niet beschikbaar')

                    # Richtingskans (Classificatie) met kleurweergave
                    prob_color = "🟢" if prob >= 50 else "🔴"
                    st.metric(
                        label=f'{prob_color} Richtingskans (Stijging)', 
                        value=f'{prob:.1f}%'
                    )

                    # Extra kolommen voor compacte weergave van parameters
                    sub1, sub2 = st.columns(2)
                    with sub1:
                        st.metric(label='Leersnelheid', value=f'{lr:.4f}')
                    with sub2:
                        st.metric(label='Lookback', value=f'{lookback}c')

                    # Risicobeheer (SL / TP)
                    if sl > 0 and tp > 0:
                        st.markdown(f"🛡️ **SL:** `${sl:,.2f}` | 🎯 **TP:** `${tp:,.2f}`")

                    # Ruwe state details uitklapmenu
                    with st.expander('📄 Model & State Details'):
                        st.json(state)
                else:
                    st.warning(
                        f'Geen data voor `{tf}`.\n\nWacht tot de worker deze heeft ingeladen.'
                    )

# Extra sectie voor systeemstatus
st.markdown('---')
st.subheader('📊 Institutionele Systeemstatus')
st.write(
    'De worker maakt gebruik van gerichte optimalisatie, dynamische modelweging '
    '(XGBoost, LightGBM, CatBoost) en ATR-gebaseerd risicobeheer per tijdsbestek.'
)
