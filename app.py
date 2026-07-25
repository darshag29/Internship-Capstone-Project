"""
Stock Market Intelligence System - Streamlit App (Task 12)

Run with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd

from src.config import ASSETS, ASSET_DISPLAY_NAMES
from src.model import run_all_assets, metrics_comparison_table, predict_next_direction
from src.business_metrics import find_optimal_threshold
from src.explainability import get_shap_values, global_feature_importance
from src.eda import plot_price_trend, plot_correlation_heatmap
from src.data_fetch import fetch_all_news
from src.data_cleaning import clean_news_df
from src.rag_pipeline import build_documents, build_index, answer_question

st.set_page_config(page_title="Stock Market Intelligence System", layout="wide")
st.title("Stock Market Intelligence System")


# ---------------------------------------------------------------------
# Cached pipeline: runs once per session (or until cache is cleared)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models for all assets (first run only)...")
def load_results():
    return run_all_assets()


@st.cache_resource(show_spinner="Fetching & indexing news for chat...")
def load_rag_index(_results):
    try:
        news_df = fetch_all_news()
        news_df = clean_news_df(news_df)
    except Exception as exc:
        st.warning(f"News fetch failed ({exc}). Chat will only know about model results.")
        news_df = pd.DataFrame(columns=["ticker", "title", "description", "publishedAt", "source"])
    docs = build_documents(news_df, _results)
    build_index(docs)
    return True


results = load_results()
if not results:
    st.error("No models could be trained. Check your internet connection / tickers in src/config.py.")
    st.stop()

rag_ready = load_rag_index(results)

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.header("Settings")
available_tickers = list(results.keys())
selected_ticker = st.sidebar.selectbox(
    "Select asset",
    available_tickers,
    format_func=lambda t: ASSET_DISPLAY_NAMES.get(t, t),
)
llm_provider = st.sidebar.radio("RAG answer generator", ["cohere", "gemini"], horizontal=True)

tab1, tab2, tab3 = st.tabs(["Predictions", "Chat", "Comparison"])

# ---------------------------------------------------------------------
# Tab 1: Predictions
# ---------------------------------------------------------------------
with tab1:
    r = results[selected_ticker]
    name = ASSET_DISPLAY_NAMES.get(selected_ticker, selected_ticker)
    st.subheader(f"{name} ({selected_ticker})")

    col1, col2, col3 = st.columns(3)
    pred = predict_next_direction(r)
    col1.metric("Predicted next-day direction", pred["direction"])
    col2.metric("Confidence", f"{pred['confidence']*100:.1f}%")
    col3.metric("Model accuracy (test set)", f"{r['metrics']['accuracy']*100:.1f}%")

    st.plotly_chart(plot_price_trend(r["feature_df"], title=f"{name} Price Trend"), use_container_width=True)

    st.markdown("#### Threshold tuning (business cost)")
    tuning = find_optimal_threshold(r["y_test"], r["y_proba"])
    tcol1, tcol2 = st.columns(2)
    tcol1.metric("Optimal threshold", tuning["optimal_threshold"])
    tcol2.metric("Cost saved vs default (0.5)", f"${tuning['savings_vs_default']:.0f}")
    st.line_chart(tuning["cost_table"].set_index("threshold")["total_cost"])

    st.markdown("#### SHAP Explainability")
    shap_values, X_scaled_df = get_shap_values(r["model"], r["X_test"])
    importance_df = global_feature_importance(shap_values)
    st.bar_chart(importance_df.set_index("feature")["mean_abs_shap"].head(10))

    st.caption("Waterfall plot for the most recent prediction:")
    import matplotlib.pyplot as plt
    import shap as shap_lib
    fig, ax = plt.subplots(figsize=(8, 5))
    shap_lib.plots.waterfall(shap_values[-1], show=False)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------------
# Tab 2: Chat (RAG)
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Ask about stocks, predictions, or recent news")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    user_question = st.chat_input(
        "e.g. Which stock has the highest accuracy? / What is the predicted direction for Reliance?"
    )
    if user_question:
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = answer_question(user_question, provider=llm_provider)
                    st.write(result["answer"])
                    with st.expander("Sources"):
                        for doc, meta in result["sources"]:
                            st.caption(f"[{meta.get('type')}] {doc}")
                    st.session_state.chat_history.append(("assistant", result["answer"]))
                except Exception as exc:
                    err = f"Sorry, I couldn't generate an answer: {exc}"
                    st.error(err)
                    st.session_state.chat_history.append(("assistant", err))

# ---------------------------------------------------------------------
# Tab 3: Comparison
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Compare all assets")
    comparison_df = metrics_comparison_table(results)
    comparison_df.index = [ASSET_DISPLAY_NAMES.get(t, t) for t in comparison_df.index]
    st.dataframe(comparison_df.style.format("{:.3f}"))

    st.markdown("#### Accuracy / F1 / AUC by asset")
    st.bar_chart(comparison_df[["accuracy", "f1", "auc"]])

    st.markdown("#### Predicted direction, all assets")
    pred_rows = []
    for ticker, r in results.items():
        p = predict_next_direction(r)
        pred_rows.append({
            "Asset": ASSET_DISPLAY_NAMES.get(ticker, ticker),
            "Prediction": p["direction"],
            "Confidence": f"{p['confidence']*100:.1f}%",
        })
    st.table(pd.DataFrame(pred_rows))

    st.markdown("#### Return correlation across assets")
    price_dict = {ASSET_DISPLAY_NAMES.get(t, t): r["feature_df"] for t, r in results.items()}
    st.plotly_chart(plot_correlation_heatmap(price_dict), use_container_width=True)
