# pdf_parser.py — Reads PDF files and splits text into overlapping chunks

import os
import re
from pypdf import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract text from each page of a PDF.
    Returns a list of dicts: { "text": str, "page": int, "source": str }
    """
    reader = PdfReader(pdf_path)
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "page": i + 1,
                "source": book_name
            })

    print(f"[pdf_parser] Extracted {len(pages)} pages from '{book_name}'")
    return pages

def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Group each page's sentences into chunks close to CHUNK_SIZE words,
    never cutting a sentence in half, with a sentence-level overlap
    between consecutive chunks so context carries across the split.
    Returns a list of dicts: { "text": str, "page": int, "source": str }
    """
    chunks = []

    for page in pages:
        sentences = _split_sentences(page["text"])
        if not sentences:
            continue

        current, current_len = [], 0
        for sentence in sentences:
            sentence_len = len(sentence.split())

            if current and current_len + sentence_len > CHUNK_SIZE:
                chunks.append({
                    "text": " ".join(current),
                    "page": page["page"],
                    "source": page["source"]
                })

                # Carry the trailing sentences forward as overlap
                overlap, overlap_len = [], 0
                for s in reversed(current):
                    s_len = len(s.split())
                    if overlap_len + s_len > CHUNK_OVERLAP:
                        break
                    overlap.insert(0, s)
                    overlap_len += s_len
                current, current_len = overlap, overlap_len

            current.append(sentence)
            current_len += sentence_len

        if current and current_len >= 10:  # skip tiny leftover chunks
            chunks.append({
                "text": " ".join(current),
                "page": page["page"],
                "source": page["source"]
            })

    print(f"[pdf_parser] Produced {len(chunks)} chunks total")
    return chunks

def load_pdf(pdf_path: str) -> list[dict]:
    """Main entry point: extract + chunk a single PDF file."""
    pages = extract_pages(pdf_path)
    return chunk_pages(pages)