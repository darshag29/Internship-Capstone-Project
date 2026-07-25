# Stock Market Intelligence System

An end-to-end system that predicts next-day stock direction, explains predictions with SHAP,
answers natural-language questions via a RAG pipeline, and compares multiple assets — all in
an interactive Streamlit app.

Covers all 12 capstone tasks: data acquisition (yfinance + NewsAPI) → cleaning → EDA →
feature engineering (18 features) → Random Forest modeling per asset → business-cost metrics
→ threshold tuning → SHAP explainability → RAG chat (ChromaDB + Cohere/Gemini) → Streamlit app.

## Project structure

```
stock_intel/
├── app.py                    # Streamlit app (Task 12) - run this
├── main.py                   # CLI pipeline runner (Tasks 1-10), no UI
├── requirements.txt
├── .env.example               # copy to .env and fill in API keys
├── data/                       # downloaded CSVs land here
├── models/                     # trained model .joblib files land here
└── src/
    ├── config.py               # assets list, paths, API keys, cost assumptions
    ├── data_fetch.py            # Task 1-2: yfinance + NewsAPI
    ├── data_cleaning.py         # Task 3
    ├── eda.py                   # Task 4: plotly charts
    ├── feature_engineering.py   # Task 5: 18 engineered features + target
    ├── model.py                 # Task 6-8: process_asset(), run_all_assets(), metrics table
    ├── business_metrics.py      # Task 9: cost-based threshold tuning
    ├── explainability.py        # Task 10: SHAP
    └── rag_pipeline.py          # Task 11: embeddings + ChromaDB + LLM answer generation
```

## Setup

1. **Install dependencies** (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

2. **Get API keys** (both have free tiers):
   - NewsAPI: https://newsapi.org/register
   - Cohere: https://dashboard.cohere.com/api-keys (used by default for RAG answers)
   - Optional: Gemini: https://aistudio.google.com/app/apikey (alternate LLM, selectable in the app sidebar)

3. **Configure keys**:
   ```bash
   cp .env.example .env
   # then edit .env and paste in your keys
   ```

4. **Choose your 5th stock** (optional): edit `ASSETS` in `src/config.py`. Defaults to
   Sensex, Reliance, TCS, Infosys, HDFC Bank.

## Run

**Full interactive app:**
```bash
streamlit run app.py
```
This will (on first run) download 5 years of data for each asset, train a Random Forest
model per asset, fetch recent news, and build the RAG index — then open three tabs:
- **Predictions** — price chart, tomorrow's UP/DOWN call, confidence, threshold tuning,
  SHAP feature importance and waterfall plot for the selected asset.
- **Chat** — ask things like *"What is the predicted direction for Reliance tomorrow?"*,
  *"Which stock has the highest accuracy?"*, *"Compare TCS and Infosys"*.
- **Comparison** — side-by-side metrics table, accuracy/F1/AUC bar chart, all-asset
  predictions, and a return-correlation heatmap.

**CLI pipeline only** (no UI, prints results to terminal — good for a first data pull or
quick debugging):
```bash
python main.py
```

## Notes

- The train/test split is **time-based** (last 20% of each asset's history), not random —
  correct for a time series so the model is never tested on data "from the past" relative
  to training.
- Business costs (`COST_FALSE_POSITIVE=100`, `COST_FALSE_NEGATIVE=50` in `src/config.py`)
  are placeholder assumptions — a false positive (bad trade taken) is assumed twice as
  costly as a false negative (good trade missed). Adjust to match your actual strategy.
- `st.cache_resource` in `app.py` means models train once per server session; delete the
  cache (or restart the app) to retrain on fresh data.
- NewsAPI's free tier only returns articles from the last month and is rate-limited —
  the chat's news knowledge is limited to that recent window.


  **AUTHOR**
  Darsh Agarwal
