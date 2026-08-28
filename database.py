# database.py — SQLite storage for document chunks and their embeddings

import re
import sqlite3
import numpy as np
from config import DB_PATH

def get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # access columns by name
    return conn

def init_db() -> None:
    """Create the documents table (and its keyword index) if missing."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            source    TEXT    NOT NULL,   -- book name
            page      INTEGER NOT NULL,   -- page number in the PDF
            content   TEXT    NOT NULL,   -- raw text chunk
            embedding BLOB    NOT NULL    -- numpy float32 bytes
        )
    """)

    # Full-text index for keyword/lexical search (hybrid search alongside
    # embedding similarity). External-content table keyed on documents.id,
    # kept in sync via triggers so callers never touch it directly.
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            content, content='documents', content_rowid='id'
        )
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, content) VALUES (new.id, new.content);
        END
    """)

    # Backfill the FTS index for rows inserted before it existed.
    fts_count = conn.execute("SELECT count(*) FROM documents_fts").fetchone()[0]
    doc_count = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
    if fts_count < doc_count:
        conn.execute("INSERT INTO documents_fts(rowid, content) SELECT id, content FROM documents "
                     "WHERE id NOT IN (SELECT rowid FROM documents_fts)")

    conn.commit()
    conn.close()
    print("[database] Initialised database at:", DB_PATH)

def source_exists(source: str) -> bool:
    """Return True if chunks for this source (book) are already stored."""
    conn = get_connection()
    row = conn.execute(
        "SELECT 1 FROM documents WHERE source = ? LIMIT 1", (source,)
    ).fetchone()
    conn.close()
    return row is not None

def insert_chunks(chunks: list[dict], embeddings: list[np.ndarray]) -> None:
    """
    Persist a list of chunks alongside their embedding vectors.
    chunks  : list of { "text", "page", "source" }
    embeddings : parallel list of numpy float32 arrays
    """
    conn = get_connection()
    rows = [
        (
            chunk["source"],
            chunk["page"],
            chunk["text"],
            emb.tobytes()           # store raw bytes
        )
        for chunk, emb in zip(chunks, embeddings)
    ]
    conn.executemany(
        "INSERT INTO documents (source, page, content, embedding) VALUES (?, ?, ?, ?)",
        rows
    )
    conn.commit()
    conn.close()
    print(f"[database] Saved {len(rows)} chunks to database")

def load_all_chunks() -> list[dict]:
    """
    Load every stored chunk with its embedding.
    Returns list of { "id", "source", "page", "content", "embedding" (ndarray) }
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, source, page, content, embedding FROM documents"
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        results.append({
            "id":      row["id"],
            "source":  row["source"],
            "page":    row["page"],
            "content": row["content"],
            "embedding": emb
        })
    return results

def keyword_search(query: str, limit: int) -> list[dict]:
    """
    Lexical search over chunk content via SQLite FTS5 (BM25 ranking).
    Good for exact terms embeddings can miss — function names, error codes,
    class names, version numbers. Returns [] if the query has no usable terms.
    """
    terms = re.findall(r"\w+", query)
    if not terms:
        return []
    match_query = " OR ".join(terms)

    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT documents.id, documents.source, documents.page, documents.content,
                   bm25(documents_fts) AS rank
            FROM documents_fts
            JOIN documents ON documents.id = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (match_query, limit)).fetchall()
    except sqlite3.OperationalError:
        # Malformed FTS query (e.g. a lone reserved character) — degrade to no keyword hits.
        rows = []
    conn.close()

    return [
        {"id": r["id"], "source": r["source"], "page": r["page"], "content": r["content"], "bm25": r["rank"]}
        for r in rows
    ]