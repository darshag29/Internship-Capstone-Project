"""
Task 11: RAG pipeline.

Embeds news articles + model results into ChromaDB, retrieves the most
relevant context for a user question, then calls Cohere (default) or
Gemini to generate a grounded answer.
"""
import uuid
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import COHERE_API_KEY, GEMINI_API_KEY, ASSET_DISPLAY_NAMES

_embedder = None
_chroma_client = None
_collection = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def build_documents(news_df: pd.DataFrame, results: dict) -> list[dict]:
    """Turn news rows + model metrics/predictions into flat text documents for retrieval."""
    docs = []

    # News articles
    for _, row in news_df.iterrows():
        text = f"[{row.get('ticker', '')}] {row.get('title', '')}. {row.get('description', '')}"
        docs.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "metadata": {
                "type": "news",
                "ticker": str(row.get("ticker", "")),
                "date": str(row.get("publishedAt", "")),
                "source": str(row.get("source", "")),
            },
        })

    # Model results (predictions + metrics) so the LLM can answer things like
    # "which stock has the highest accuracy" or "predicted direction for Reliance"
    for ticker, r in results.items():
        name = ASSET_DISPLAY_NAMES.get(ticker, ticker)
        m = r["metrics"]
        text = (
            f"Model results for {name} ({ticker}): accuracy={m['accuracy']:.3f}, "
            f"precision={m['precision']:.3f}, recall={m['recall']:.3f}, "
            f"f1={m['f1']:.3f}, auc={m['auc']:.3f}."
        )
        docs.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "metadata": {"type": "model_result", "ticker": ticker, "date": ""},
        })

    return docs


def build_index(documents: list[dict], collection_name: str = "stock_intel"):
    """Task 11: create embeddings and store them in a fresh ChromaDB collection."""
    global _chroma_client, _collection
    _chroma_client = chromadb.Client()

    # Recreate the collection each run so re-indexing doesn't duplicate docs
    try:
        _chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    _collection = _chroma_client.create_collection(collection_name)

    if not documents:
        return _collection

    embedder = get_embedder()
    texts = [d["text"] for d in documents]
    embeddings = embedder.encode(texts).tolist()

    _collection.add(
        ids=[d["id"] for d in documents],
        embeddings=embeddings,
        documents=texts,
        metadatas=[d["metadata"] for d in documents],
    )
    return _collection


def retrieve(query: str, top_k: int = 5):
    """Retrieve the top_k most relevant documents for `query`."""
    if _collection is None:
        raise RuntimeError("Call build_index() before retrieve().")
    embedder = get_embedder()
    query_embedding = embedder.encode([query]).tolist()
    hits = _collection.query(query_embeddings=query_embedding, n_results=top_k)
    return list(zip(hits["documents"][0], hits["metadatas"][0]))


def generate_answer(query: str, context_docs: list[tuple], provider: str = "cohere") -> str:
    """Call an LLM with the retrieved context to produce a grounded answer."""
    context_text = "\n".join(f"- {doc}" for doc, _meta in context_docs)
    prompt = (
        "You are a financial analyst assistant. Answer the user's question "
        "using ONLY the context below. If the context doesn't contain the "
        "answer, say so honestly.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    )

    if provider == "cohere":
        import cohere
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY not set in .env")
        co = cohere.Client(COHERE_API_KEY)
        resp = co.chat(message=prompt, model="command-r")
        return resp.text

    elif provider == "gemini":
        import google.generativeai as genai
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return resp.text

    else:
        raise ValueError(f"Unknown provider: {provider}")


def answer_question(query: str, top_k: int = 5, provider: str = "cohere") -> dict:
    """Full RAG call: retrieve context, then generate an answer."""
    context_docs = retrieve(query, top_k=top_k)
    answer = generate_answer(query, context_docs, provider=provider)
    return {"answer": answer, "sources": context_docs}
