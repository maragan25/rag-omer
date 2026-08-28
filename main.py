# main.py — Entry point: ingest PDFs and run the study Q&A chat loop

import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from database import init_db, source_exists, insert_chunks
from pdf_parser import load_pdf
from embeddings import get_embeddings_batch
from retriever import load_cache, find_relevant_chunks, format_context
from llm import generate_answer
from config import CONFIDENCE_THRESHOLD, HISTORY_TURNS

def ingest(pdf_path: str) -> None:
    """Parse a PDF, embed its chunks, and store them in SQLite."""
    if not os.path.exists(pdf_path):
        print(f"[main] File not found: {pdf_path}")
        return

    book_name = os.path.splitext(os.path.basename(pdf_path))[0]

    if source_exists(book_name):
        print(f"[main] '{book_name}' is already ingested. Skipping.")
        return

    print(f"\n[main] Ingesting: {pdf_path}")
    chunks     = load_pdf(pdf_path)
    texts      = [c["text"] for c in chunks]
    embeddings = get_embeddings_batch(texts)
    insert_chunks(chunks, embeddings)
    print(f"[main] '{book_name}' ready for querying.\n")

def chat_loop() -> None:
    """Interactive Q&A loop — type 'quit' or 'exit' to stop."""
    print("\nStudy RAG Assistant (Foundry Local)")
    print("Type your question, or 'quit' to exit.\n")

    history: list[tuple[str, str]] = []  # recent (question, answer) turns, oldest first

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        # Retrieve → Augment → Generate
        chunks  = find_relevant_chunks(question)
        if not chunks:
            print("Assistant: No relevant content found. Try ingesting more PDFs.\n")
            continue

        if chunks[0]["score"] < CONFIDENCE_THRESHOLD:
            print("[main] Note: the best match for this question is a weak one — "
                  "try asking something more specific if the answer looks off.\n")

        context = format_context(chunks)
        answer  = generate_answer(question, context, history=history)

        history.append((question, answer))
        history = history[-HISTORY_TURNS:]

        print(f"\nAssistant: {answer}\n")
        print("-" * 60)

if __name__ == "__main__":
    # Initialise DB
    init_db()

    # Accept PDF paths as command-line args:  python main.py book1.pdf book2.pdf
    pdf_args = [arg for arg in sys.argv[1:] if arg.endswith(".pdf")]

    if pdf_args:
        for pdf in pdf_args:
            ingest(pdf)
    else:
        print("[main] No PDFs provided. Usage: python main.py book1.pdf book2.pdf ...")
        print("[main] Starting chat with existing database content...\n")

    # Load every chunk's embedding into memory once, so queries never hit disk.
    load_cache()

    chat_loop()