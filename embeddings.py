# embeddings.py — Generates text embeddings using Foundry Local SDK

import numpy as np
from foundry_client import get_ready_model
from config import EMBEDDING_MODEL

_client = get_ready_model(EMBEDDING_MODEL).get_embedding_client()

def get_embedding(text: str) -> np.ndarray:
    """
    Generate a single embedding vector for a given text string.
    Returns a normalised numpy float32 array.
    """
    response = _client.generate_embedding(text)

    vector = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector

def get_query_embedding(query: str) -> np.ndarray:
    """
    Embed a *search query* (not a stored passage).

    qwen3-embedding is instruction-tuned and expects asymmetric retrieval:
    passages are embedded as-is (see insert path), but queries should carry a
    short task instruction. Adding it here measurably improves recall without
    touching any stored vector.
    """
    task = "Given a question from a student, retrieve textbook passages that answer it"
    return get_embedding(f"Instruct: {task}\nQuery: {query}")

def get_embeddings_batch(texts: list[str]) -> list[np.ndarray]:
    """
    Generate embeddings for a list of texts.
    Processes in one API call where possible; falls back to one-by-one.
    """
    try:

        response = _client.generate_embeddings(texts)

        vectors = [
            np.array(item.embedding, dtype=np.float32)
            for item in response.data
        ]
    except Exception:
        vectors = [get_embedding(t) for t in texts]

    normalised = []
    for v in vectors:
        norm = np.linalg.norm(v)
        normalised.append(v / norm if norm > 0 else v)
    return normalised