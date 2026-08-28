# eval.py — Lightweight evaluation harness for the RAG pipeline
#
# Fill in TEST_CASES with questions you know the answer to (and which page of
# which ingested book it lives on), then run:  python eval.py
#
# It automatically checks retrieval (did the expected page show up in the
# top-K?) and reports timing/confidence — answer quality still needs a human
# glance at the printed text, this doesn't grade that part for you.

import time
from database import init_db
from retriever import load_cache, find_relevant_chunks, format_context
from llm import generate_answer

# expected_page is the PDF page index (as stored in the DB), not the printed
# page number. The C-book anchors below were verified against the ingested DB.
# Re-chunking (CHUNK_SIZE change) does not move page numbers, so these stay valid
# after a re-ingest. Add your own Thomas_Calculus cases once ingested — note its
# text extraction is weaker (math symbols), so retrieval there is less reliable.
C_BOOK = "The.C.Programming.Language.2Nd.Ed Prentice.Hall.Brian.W.Kernighan.and.Dennis.M.Ritchie"

TEST_CASES = [
    {"question": "What are the variables named in a structure called?",
     "expected_source": C_BOOK, "expected_page": 115},
    {"question": "To what kinds of variables can the register declaration be applied?",
     "expected_source": C_BOOK, "expected_page": 76},
    {"question": "How is an unbuffered single-character getchar implemented with read?",
     "expected_source": C_BOOK, "expected_page": 153},
    {"question": "When does a local (automatic) variable come into existence?",
     "expected_source": C_BOOK, "expected_page": 32},
    {"question": "Can a pointer be converted to type void * and back without change?",
     "expected_source": C_BOOK, "expected_page": 175},
]

def run_evaluation() -> None:
    if not TEST_CASES:
        print("[eval] TEST_CASES is empty — add some question/expected-source/expected-page "
              "entries at the top of eval.py first.")
        return

    init_db()
    load_cache()

    results = []
    for case in TEST_CASES:
        start = time.time()
        chunks = find_relevant_chunks(case["question"])
        retrieval_hit = any(
            c["source"] == case["expected_source"] and c["page"] == case["expected_page"]
            for c in chunks
        )
        answer = generate_answer(case["question"], format_context(chunks)) if chunks else ""
        elapsed = time.time() - start

        results.append({
            "question":      case["question"],
            "retrieval_hit": retrieval_hit,
            "top_score":     round(chunks[0]["score"], 3) if chunks else None,
            "latency_sec":   round(elapsed, 2),
            "answer":        answer
        })

    hits         = sum(r["retrieval_hit"] for r in results)
    scored       = [r["top_score"] for r in results if r["top_score"] is not None]
    avg_score    = sum(scored) / len(scored) if scored else 0.0
    avg_latency  = sum(r["latency_sec"] for r in results) / len(results)

    print("\n=== Evaluation Report ===")
    for r in results:
        status = "HIT " if r["retrieval_hit"] else "MISS"
        print(f"[{status}] score={r['top_score']} {r['latency_sec']}s — {r['question']}")
        print(f"        answer: {r['answer'][:150]}{'...' if len(r['answer']) > 150 else ''}")

    print("\n--- Summary ---")
    print(f"Retrieval accuracy: {hits}/{len(results)} ({100 * hits / len(results):.0f}%)")
    print(f"Average top score : {avg_score:.3f}")
    print(f"Average latency   : {avg_latency:.2f}s")

if __name__ == "__main__":
    run_evaluation()
