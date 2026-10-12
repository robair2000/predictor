# ⚡ RV Multi-Timeframe Ensemble Predictor & Quant Dashboard

Een institutioneel, zelflerend kwantitatief handelssysteem en real-time Streamlit-dashboard voor cryptocurrency-analyse. Het systeem maakt gebruik van een krachtig AI-ensemble, dynamische modelweging, multi-timeframe ondersteuning en geautomatiseerd risicobeheer.

---

## 🚀 Over het Project
Dit project is geëvolueerd van een simpel lineair regressiemodel voor één koers naar een volwaardig, adaptief kwantitatief platform. De achtergrond-worker draait autonoom op Render, traint en optimaliseert continu machine learning-modellen per tijdsbestek, en synchroniseert de status via GitHub met een interactief webdashboard.

### Ondersteunde Assets
* **Bitcoin (BTC)**
* **Ethereum (ETH)**
* **Chainlink (LINK)**

---

## 📊 Belangrijkste Kenmerken

### 1. Multi-Timeframe Ondersteuning
De app ondersteunt 5 verschillende resoluties om zowel intraday als macro-economische marktbewegingen te analyseren:
* **15 Minuten (`15m`)**
* **1 Uur (`1h`)**
* **1 Dag (`1d`)**
* **1 Week (`1w`)**
* **1 Maand (`1m`)**

### 2. Geavanceerd AI-Ensemble & Modelweging
In plaats van één enkel model maakt het systeem gebruik van een gewogen ensemble van drie top-tier machine learning-frameworks:
* **XGBoost** (Regressie & Classificatie)
* **LightGBM**
* **CatBoost**
* **Dynamische Weging:** De worker houdt de voorspellingsfout per model bij en kent automatisch meer gewicht toe aan het model dat op dat moment het meest accuraat presteert.
* **Richtingskans (Direction Probability):** Een classificatiemodel berekent de exacte waarschijnlijkheid (in %) of de koers in de volgende periode zal stijgen of dalen.

### 3. Technische Indicatoren & Features
De modellen worden gevoed met een robuuste set technische indicatoren:
* **Trend & Momentum:** SMA (20), EMA (9/21), MACD en RSI (14).
* **Volatiliteit & Volume:** ATR (Average True Range), Bollinger Bands, OBV (On-Balance Volume) en VWAP.

### 4. Adaptief Leren & Risicobeheer
* **Zelfoptimalisatie:** De bot past autonoom zijn leersnelheid (`learning_rate`) en historische trainingsvenster (`lookback_window` tussen 60 en 300 candles) aan op basis van eerdere voorspellingsfouten.
* **Automatisch Risicobeheer:** Op basis van de actuele ATR berekent de worker concrete **Stop-Loss (SL)** en **Take-Profit (TP)** richtprijzen.

---

## 🏗️ Systeemarchitectuur

1. **Achtergrond-Worker (`quant_worker.py`):** 
   * Draait op Render en haalt periodiek live marktdata op via `yfinance`.
   * Voert feature engineering en modeltraining uit per asset en tijdsbestek.
   * Slaat de resultaten op in JSON-bestanden (`model_state_*.json`) en pusht deze automatisch naar GitHub.
2. **Dashboard (`app.py`):** 
   * Een interactieve webinterface gebouwd met **Streamlit**.
   * Leest de JSON-statusbestanden uit, haalt live koersdata op en visualiseert prijsverlopen, technische indicatoren en modelparameters.

---

## 🛠️ Installatie & Lokaal Draaien

### 1. Kloon de Repository
```bash
git clone [https://github.com/robair2000/predictor.git](https://github.com/robair2000/predictor.git)
cd predictor
