# config.py — Central settings for the Study RAG Assistant

# --- Foundry Local Models ---
EMBEDDING_MODEL = "qwen3-embedding-0.6b"   # Embedding model alias
LLM_MODEL       = "phi-3.5-mini"           # Chat/completion model alias

# --- Chunking ---
# Kept small on purpose: a 500-word chunk is mostly text unrelated to any one
# question, which both blurs retrieval ranking and dilutes the answer. ~200-word
# chunks retrieve more precisely, and several of them still fit under the chat
# model's prefill budget (see TOP_K). Changing these requires a re-ingest:
#   del study_rag.sqlite  &&  python main.py *.pdf
CHUNK_SIZE    = 200   # Max words per chunk
CHUNK_OVERLAP = 40    # Overlapping words between chunks

# --- Retrieval ---
TOP_K = 3  # Number of chunks to retrieve per query.
           # Measured empirically on this machine: Foundry Local's chat
           # completion has an internal watchdog that cancels a request if
           # it runs too long, and CPU-only prefill on phi-3.5-mini scales
           # badly with context size — ~1000 words of context reliably
           # finished in ~60s, while ~1500-2000 words got cancelled around
           # 120-190s. With ~200-word chunks, 3 chunks (~600 words) stays
           # comfortably inside that budget while giving broader coverage
           # than the old 2x500. Reducing max_tokens didn't help, confirming
           # prefill (not generation length) is the bottleneck. If you switch
           # to a smaller/faster chat model or get GPU acceleration working,
           # this can go higher.
CONFIDENCE_THRESHOLD = 0.55  # Below this top semantic score, the match is weak — warn the student

# Hybrid search weighting: how much retrieval ranking trusts embedding
# similarity (semantic) vs. BM25 keyword matching (lexical). Keyword matching
# matters more for CS material — exact function/class names, error codes,
# version numbers — that embeddings can blur together.
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT  = 0.3

# --- Conversation ---
HISTORY_TURNS = 4  # Past Q/A pairs kept in context so follow-up questions work

# --- Database ---
DB_PATH = "study_rag.sqlite"  # SQLite database file path

# --- System Prompt ---
SYSTEM_PROMPT = """You are a helpful study assistant.
Answer the user's question using ONLY the context provided below — never rely on outside knowledge.

Rules:
- If the context does not contain the answer, say exactly: "I don't have that information in my notes."
- Never infer or guess information that isn't explicitly stated in the context.
- Treat each numbered context passage as possibly coming from a different part of the book — don't blend unrelated passages into a single claim.
- If passages disagree or describe different things, point out the discrepancy instead of silently picking one.
- Prefer quoting short key phrases from the context over paraphrasing when precision matters (definitions, formulas, function signatures, error messages).
- Lead with the direct answer in the first sentence. Do not restate the question or narrate your process ("Based on the context...").
- Keep the answer to 2-4 sentences unless the question genuinely needs more.

End every answer with a line "Sources:" followed by each passage you used, formatted as "- <book>, page <page>".
"""