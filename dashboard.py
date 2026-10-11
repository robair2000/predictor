import json
import os
import streamlit as st

# Pagina configuratie
st.set_page_config(
    page_title='RV dashboard', page_icon='📈', layout='wide'
)

st.title('🚀 RV prediction dashboard')
st.markdown(
    'Real-time overzicht van je op Render draaiende XGBoost kwantitatieve worker.'
)

# Assets die je beheert
assets = {'BTC': 'BTC', 'ETH': 'ETH', 'LINK': 'LINK', 'AAVE': 'AAVE'}


def load_state(asset_name):
  state_file = f'model_state_{asset_name.lower()}.json'
  if os.path.exists(state_file):
    with open(state_file, 'r') as f:
      return json.load(f)
  return None


# Sidebar navigatie of info
st.sidebar.header('Besturing')
if st.sidebar.button('🔄 Gegevens verversen'):
  st.rerun()

st.sidebar.markdown('---')
st.sidebar.info(
    'Dit dashboard leest de JSON state-bestanden uit die door je Render worker'
    ' worden bijgewerkt.'
)

# Layout in kolommen per asset
cols = st.columns(len(assets))

for i, (display_name, key) in enumerate(assets.items()):
  with cols[i]:
    st.subheader(f'🪙 {display_name}')
    state = load_state(key)

    if state:
      # Toon belangrijkste metrics
      last_pred = state.get('last_prediction')
      lr = state.get('learning_rate')
      baseline_err = state.get('baseline_error')

      if last_pred is not None:
        st.metric(
            label='Laatste Voorspelling', value=f'$ {last_pred:,.2f}'
        )
      else:
        st.metric(label='Laatste Voorspelling', value='Nog niet beschikbaar')

      st.metric(label='Leersnelheid (LR)', value=f'{lr:.4f}')

      if baseline_err != float('inf'):
        st.metric(label='Baseline Fout', value=f'{baseline_err:.2f}')
      else:
        st.metric(label='Baseline Fout', value='Initieel')

      with st.expander('📄 Ruwe State Details'):
        st.json(state)
    else:
      st.warning(
          f'Geen state-bestand gevonden (`model_state_{key.lower()}.json`).'
          ' Wacht tot de worker minimaal één keer heeft gedraaid.'
      )

# Extra sectie voor prestatieanalyse
st.markdown('---')
st.subheader('📊 Systeemstatus & Optimalisatie')
st.write(
    'De worker past automatisch de hyperparameters van de XGBoost-modellen aan'
    ' op basis van de absolute voorspellingsfout van de vorige 15-minuten run.'
)
