# src/sparse_retriever.py

import os
import json
import re
import pickle
from rank_bm25 import BM25Okapi


def load_chunks(json_path: str) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list:
    # Simple, language-agnostic tokenizer: lowercase, split on non-alphanumerics
    return re.findall(r"\w+", text.lower())


def build_bm25(chunks: list) -> BM25Okapi:
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    return BM25Okapi(tokenized_corpus)


def save_bm25(bm25: BM25Okapi, chunks: list, output_path: str) -> None:
    with open(output_path, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"BM25 index saved to {output_path}")


class SparseRetriever:
    """Keyword-based retrieval using BM25 — catches exact term matches
    (like 'Section 177' or '₹1000') that semantic search can miss or
    under-rank."""

    def __init__(self, bm25_path: str):
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]

    def query(self, question: str, top_k: int = 10) -> list:
        tokenized_query = tokenize(question)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["sparse_score"] = float(scores[idx])
            results.append(chunk)
        return results


if __name__ == "__main__":
    chunk_path = os.path.join("..", "data", "processed", "all_chunks.json")
    bm25_path = os.path.join("..", "data", "processed", "bm25_index.pkl")

    chunks = load_chunks(chunk_path)
    bm25 = build_bm25(chunks)
    save_bm25(bm25, chunks, bm25_path)