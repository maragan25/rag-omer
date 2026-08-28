# retriever.py — Finds the most relevant chunks for a user query
#
# Uses an in-memory cache of every chunk's embedding (loaded once, refreshed
# after ingestion) so a query never touches disk, and scores chunks with a
# hybrid of embedding similarity (semantic) and BM25 keyword matching
# (lexical) — the latter catches exact terms embeddings tend to miss, like
# function names, error codes, or version numbers in CS reference material.

import numpy as np
from embeddings import get_query_embedding
from database import load_all_chunks, keyword_search
from config import TOP_K, SEMANTIC_WEIGHT, KEYWORD_WEIGHT

_cache = {"chunks": [], "matrix": None}

def load_cache() -> None:
    """Load every chunk + embedding from SQLite into memory. Call once at startup."""
    chunks = load_all_chunks()
    _cache["chunks"] = chunks
    _cache["matrix"] = (
        np.vstack([c["embedding"] for c in chunks]) if chunks else np.empty((0, 0), dtype=np.float32)
    )
    print(f"[retriever] Cached {len(chunks)} chunks in memory")

def refresh_cache() -> None:
    """Re-read the database. Call after ingesting a new document."""
    load_cache()

def _semantic_scores(query_embedding: np.ndarray) -> list[float]:
    """
    Cosine similarity of the query against every cached chunk at once.
    Vectors are pre-normalised at embedding time, so a single matrix-vector
    product gives cosine similarity for the whole cache — no Python loop.
    """
    matrix = _cache["matrix"]
    if matrix.size == 0:
        return []
    return (matrix @ query_embedding).tolist()

def _normalise(values: list[float]) -> list[float]:
    """Min-max scale to [0, 1] so semantic and keyword scores are comparable."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]

def find_relevant_chunks(query: str) -> list[dict]:
    """
    Hybrid search: blend semantic (embedding) similarity with lexical (BM25)
    matching, then return the top-K chunks.
    Returns list of { "source", "page", "content", "score", "hybrid_score" }
    sorted by hybrid_score desc. "score" is the pure semantic similarity —
    used elsewhere to flag weak/uncertain matches.
    """
    chunks = _cache["chunks"]
    if not chunks:
        print("[retriever] No chunks in memory. Please ingest a PDF first.")
        return []

    query_embedding = get_query_embedding(query)
    semantic_scores = _semantic_scores(query_embedding)

    # BM25 keyword candidates. sqlite's bm25() is "smaller/more negative =
    # better", so flip the sign before normalising to the usual "higher = better".
    keyword_hits = {row["id"]: -row["bm25"] for row in keyword_search(query, limit=TOP_K * 4)}
    raw_keyword = [keyword_hits.get(c["id"]) for c in chunks]
    normalised = iter(_normalise([v for v in raw_keyword if v is not None]))
    keyword_scores = [next(normalised) if v is not None else 0.0 for v in raw_keyword]

    scored = []
    for chunk, sem_score, key_score in zip(chunks, semantic_scores, keyword_scores):
        scored.append({
            "source":      chunk["source"],
            "page":        chunk["page"],
            "content":     chunk["content"],
            "score":       sem_score,
            "hybrid_score": SEMANTIC_WEIGHT * sem_score + KEYWORD_WEIGHT * key_score
        })

    scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top = scored[:TOP_K]

    print(f"[retriever] Top {TOP_K} chunks retrieved "
          f"(hybrid scores: {[round(c['hybrid_score'], 3) for c in top]})")
    return top

def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a single context string for the LLM prompt.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Source: {chunk['source']}, Page {chunk['page']}\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(parts)
