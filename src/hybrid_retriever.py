# src/hybrid_retriever.py

import os
from retriever import Retriever
from sparse_retriever import SparseRetriever


class HybridRetriever:
    """
    Combines dense (semantic/FAISS) + sparse (BM25 keyword) retrieval
    using Reciprocal Rank Fusion (RRF). RRF just uses each result's RANK
    in its own list rather than raw scores, which avoids having to
    normalize cosine similarity against BM25 scores (very different scales).
    """

    def __init__(self, dense_index_path: str, sparse_index_path: str, chunk_json_path: str, rrf_k: int = 60):
        self.dense = Retriever(index_path=dense_index_path, chunk_json_path=chunk_json_path)
        self.sparse = SparseRetriever(bm25_path=sparse_index_path)
        self.rrf_k = rrf_k

    def query(self, question: str, top_k: int = 10, fetch_k: int = 25) -> list:
        dense_results = self.dense.query(question, top_k=fetch_k)
        sparse_results = self.sparse.query(question, top_k=fetch_k)

        rrf_scores = {}
        chunk_lookup = {}

        for rank, chunk in enumerate(dense_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (self.rrf_k + rank + 1)
            chunk_lookup[cid] = chunk

        for rank, chunk in enumerate(sparse_results):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (self.rrf_k + rank + 1)
            if cid not in chunk_lookup:
                chunk_lookup[cid] = chunk

        ranked_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        results = []
        for cid in ranked_ids[:top_k]:
            chunk = dict(chunk_lookup[cid])
            chunk["rrf_score"] = rrf_scores[cid]
            results.append(chunk)

        return results


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    retriever = HybridRetriever(
        dense_index_path=os.path.join(BASE_DIR, "data", "processed", "faiss_cosine_index.idx"),
        sparse_index_path=os.path.join(BASE_DIR, "data", "processed", "bm25_index.pkl"),
        chunk_json_path=os.path.join(BASE_DIR, "data", "processed", "all_chunks.json"),
    )

    results = retriever.query("What is the penalty for not wearing a helmet?", top_k=10)
    for r in results:
        print(f"[chunk {r['chunk_id']} | source {r['source']} | rrf {r['rrf_score']:.4f}]")
        print(r["text"][:150])
        print("-" * 60)