"""
scripts/evaluate_rag.py

Full RAG Evaluation Suite for Delhi Traffic Rules Assistant
Metrics computed:
  Retrieval:   Precision@K, Recall@K, MRR (Mean Reciprocal Rank)
  RAG Quality: Faithfulness, Answer Relevance, Context Relevance

Usage:
  # Run evaluation and save results (first time = "before" baseline):
  python scripts/evaluate_rag.py

  # Run again after making changes to compare:
  python scripts/evaluate_rag.py

  Results are saved to: scripts/eval_results/
  A before vs after comparison table is printed automatically.
"""

import sys
import os
import json
import time
import math
import datetime

# ── path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(BASE_DIR, "src"))

from generator import Generator

# ── output folder ────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(BASE_DIR, "scripts", "eval_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_SET_PATH = os.path.join(BASE_DIR, "scripts", "eval_set.json")
TOP_K = 10          # chunks retrieved per query


# ╔══════════════════════════════════════════════════════════════════╗
# ║              METRIC CALCULATION FUNCTIONS                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def precision_at_k(retrieved_sources: list[str], relevant_sources: list[str], k: int) -> float:
    """Precision@K: of the K retrieved chunks, what fraction are relevant?"""
    retrieved_at_k = retrieved_sources[:k]
    if not retrieved_at_k:
        return 0.0
    relevant_hits = sum(
        1 for src in retrieved_at_k
        if any(rel.lower() in src.lower() or src.lower() in rel.lower()
               for rel in relevant_sources)
    )
    return round(relevant_hits / len(retrieved_at_k), 4)


def recall_at_k(retrieved_sources: list[str], relevant_sources: list[str], k: int) -> float:
    """Recall@K: of all relevant sources, what fraction were retrieved in top K?"""
    if not relevant_sources:
        return 0.0
    retrieved_at_k = retrieved_sources[:k]
    found = set()
    for src in retrieved_at_k:
        for rel in relevant_sources:
            if rel.lower() in src.lower() or src.lower() in rel.lower():
                found.add(rel)
    return round(len(found) / len(relevant_sources), 4)


def mean_reciprocal_rank(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """MRR: 1 / (rank of first relevant result). 0 if no relevant result found."""
    for rank, src in enumerate(retrieved_sources, start=1):
        if any(rel.lower() in src.lower() or src.lower() in rel.lower()
               for rel in relevant_sources):
            return round(1.0 / rank, 4)
    return 0.0


def faithfulness_score(answer: str, context: str) -> float:
    """
    Heuristic faithfulness: checks what fraction of answer sentences
    contain at least one word that appears in the retrieved context.
    Range: 0.0 (hallucinated) → 1.0 (fully grounded).
    """
    if not context or not answer:
        return 0.0

    context_words = set(context.lower().split())
    sentences = [s.strip() for s in answer.replace("\n", ". ").split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    grounded = 0
    for sentence in sentences:
        words = set(sentence.lower().split())
        # A sentence is grounded if it shares at least 3 content words with the context
        overlap = len(words.intersection(context_words))
        if overlap >= 3:
            grounded += 1

    return round(grounded / len(sentences), 4)


def answer_relevance_score(query: str, answer: str) -> float:
    """
    Heuristic answer relevance: checks keyword overlap between query and answer.
    Range: 0.0 (irrelevant) → 1.0 (highly relevant).
    """
    if not query or not answer:
        return 0.0

    # Filter out very common stop words
    stop_words = {"what", "is", "the", "are", "for", "in", "of", "a", "an",
                  "on", "at", "to", "by", "and", "or", "with", "how", "can",
                  "my", "i", "do", "does", "if", "be", "will", "that", "this"}

    query_words = {w.lower() for w in query.split() if w.lower() not in stop_words and len(w) > 2}
    answer_words = {w.lower() for w in answer.split() if w.lower() not in stop_words and len(w) > 2}

    if not query_words:
        return 0.0

    overlap = len(query_words.intersection(answer_words))
    return round(min(1.0, overlap / len(query_words)), 4)


def context_relevance_score(query: str, context: str) -> float:
    """
    Heuristic context relevance: checks what fraction of the query keywords
    appear in the retrieved context.
    Range: 0.0 (irrelevant) → 1.0 (highly relevant).
    """
    if not query or not context:
        return 0.0

    stop_words = {"what", "is", "the", "are", "for", "in", "of", "a", "an",
                  "on", "at", "to", "by", "and", "or", "with", "how", "can",
                  "my", "i", "do", "does", "if", "be", "will", "that", "this"}

    query_words = {w.lower() for w in query.split() if w.lower() not in stop_words and len(w) > 2}
    context_words = set(context.lower().split())

    if not query_words:
        return 0.0

    found = sum(1 for w in query_words if w in context_words)
    return round(found / len(query_words), 4)


def keyword_hit_rate(answer: str, context: str, expected_keywords: list[str]) -> float:
    """Checks how many expected keywords appear in either the answer or context."""
    if not expected_keywords:
        return 0.0
    combined = (answer + " " + context).lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return round(hits / len(expected_keywords), 4)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    MAIN EVALUATION RUNNER                        ║
# ╚══════════════════════════════════════════════════════════════════╝

def run_evaluation() -> dict:
    print("=" * 72)
    print("🚦  DELHI TRAFFIC RULES RAG — FULL EVALUATION SUITE")
    print("=" * 72)

    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    print(f"Loaded {len(eval_set)} test cases from eval_set.json")
    print("Initializing RAG Generator (Pinecone + Groq)…")
    generator = Generator()
    print("Generator ready.\n")

    all_results = []

    # ── column headers ──────────────────────────────────────────────
    header = (
        f"{'ID':<3} │ {'P@K':>5} │ {'R@K':>5} │ {'MRR':>5} │ "
        f"{'Faith':>5} │ {'AnsRel':>6} │ {'CtxRel':>6} │ "
        f"{'KwHit':>5} │ {'Latency':>8}"
    )
    print(header)
    print("─" * len(header))

    for item in eval_set:
        q_id        = item["id"]
        query       = item["query"]
        rel_sources = item["relevant_sources"]
        exp_kws     = item["expected_answer_keywords"]

        # ── call the full RAG pipeline ──────────────────────────────
        t0  = time.time()
        res = generator.ask(query, top_k=TOP_K)
        latency_ms = (time.time() - t0) * 1000

        answer  = res.get("answer",  "")
        context = res.get("context", "")
        metrics = res.get("metrics", {}) or {}

        # ── extract retrieved sources from metrics ──────────────────
        retrieved_chunks  = metrics.get("retrieved_chunks", [])
        retrieved_sources = [c.get("source", "") for c in retrieved_chunks]

        # ── retrieval metrics ───────────────────────────────────────
        p_at_k = precision_at_k(retrieved_sources, rel_sources, TOP_K)
        r_at_k = recall_at_k(retrieved_sources, rel_sources, TOP_K)
        mrr    = mean_reciprocal_rank(retrieved_sources, rel_sources)

        # ── RAG quality metrics ─────────────────────────────────────
        faith   = faithfulness_score(answer, context)
        ans_rel = answer_relevance_score(query, answer)
        ctx_rel = context_relevance_score(query, context)
        kw_hit  = keyword_hit_rate(answer, context, exp_kws)

        short_q = (query[:28] + "…") if len(query) > 30 else query
        print(
            f"{q_id:<3} │ {p_at_k:>5.2f} │ {r_at_k:>5.2f} │ {mrr:>5.2f} │ "
            f"{faith:>5.2f} │ {ans_rel:>6.2f} │ {ctx_rel:>6.2f} │ "
            f"{kw_hit:>5.2f} │ {latency_ms:>6.0f} ms"
        )

        all_results.append({
            "id":              q_id,
            "query":           query,
            "precision_at_k":  p_at_k,
            "recall_at_k":     r_at_k,
            "mrr":             mrr,
            "faithfulness":    faith,
            "answer_relevance":ans_rel,
            "context_relevance":ctx_rel,
            "keyword_hit_rate":kw_hit,
            "latency_ms":      round(latency_ms, 1),
            "total_tokens":    metrics.get("total_tokens", 0),
            "estimated_cost":  metrics.get("estimated_cost_usd", 0.0),
            "answer_snippet":  answer[:200]
        })

    # ── aggregate ────────────────────────────────────────────────────
    n = len(all_results)
    def avg(key): return round(sum(r[key] for r in all_results) / n, 4)

    summary = {
        "timestamp":         datetime.datetime.now().isoformat(),
        "num_test_cases":    n,
        "top_k":             TOP_K,
        "avg_precision_at_k":avg("precision_at_k"),
        "avg_recall_at_k":   avg("recall_at_k"),
        "avg_mrr":           avg("mrr"),
        "avg_faithfulness":  avg("faithfulness"),
        "avg_answer_relevance": avg("answer_relevance"),
        "avg_context_relevance":avg("context_relevance"),
        "avg_keyword_hit_rate": avg("keyword_hit_rate"),
        "avg_latency_ms":    avg("latency_ms"),
        "total_cost_usd":    round(sum(r["estimated_cost"] for r in all_results), 6),
        "per_query":         all_results
    }

    # ── print summary ─────────────────────────────────────────────────
    print("=" * 72)
    print("📊  EVALUATION SUMMARY")
    print("=" * 72)
    print(f"  Test Cases          : {n}")
    print(f"  Top-K               : {TOP_K}")
    print()
    print("  ── Retrieval ──────────────────────────────────────────────")
    print(f"  Precision@{TOP_K:<2}        : {summary['avg_precision_at_k']:.4f}  ({summary['avg_precision_at_k']*100:.1f}%)")
    print(f"  Recall@{TOP_K:<2}           : {summary['avg_recall_at_k']:.4f}  ({summary['avg_recall_at_k']*100:.1f}%)")
    print(f"  MRR                 : {summary['avg_mrr']:.4f}")
    print()
    print("  ── RAG Quality ────────────────────────────────────────────")
    print(f"  Faithfulness        : {summary['avg_faithfulness']:.4f}  ({summary['avg_faithfulness']*100:.1f}%)")
    print(f"  Answer Relevance    : {summary['avg_answer_relevance']:.4f}  ({summary['avg_answer_relevance']*100:.1f}%)")
    print(f"  Context Relevance   : {summary['avg_context_relevance']:.4f}  ({summary['avg_context_relevance']*100:.1f}%)")
    print(f"  Keyword Hit Rate    : {summary['avg_keyword_hit_rate']:.4f}  ({summary['avg_keyword_hit_rate']*100:.1f}%)")
    print()
    print("  ── Performance ────────────────────────────────────────────")
    print(f"  Avg Latency         : {summary['avg_latency_ms']:.1f} ms")
    print(f"  Total Suite Cost    : ${summary['total_cost_usd']:.6f} USD")
    print("=" * 72)

    return summary


# ╔══════════════════════════════════════════════════════════════════╗
# ║              BEFORE vs AFTER COMPARISON                          ║
# ╚══════════════════════════════════════════════════════════════════╝

COMPARE_METRICS = [
    ("avg_precision_at_k",    "Precision@K"),
    ("avg_recall_at_k",       "Recall@K"),
    ("avg_mrr",               "MRR"),
    ("avg_faithfulness",      "Faithfulness"),
    ("avg_answer_relevance",  "Answer Relevance"),
    ("avg_context_relevance", "Context Relevance"),
    ("avg_keyword_hit_rate",  "Keyword Hit Rate"),
    ("avg_latency_ms",        "Avg Latency (ms)"),
    ("total_cost_usd",        "Total Cost (USD)"),
]


def save_and_compare(new_summary: dict):
    """Save results and print a before-vs-after diff if a previous run exists."""
    result_files = sorted([
        f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")
    ])

    # Save current run
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"eval_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_summary, f, indent=2, ensure_ascii=False)
    print(f"\n💾  Results saved → {out_path}")

    if not result_files:
        print("\n  (This is your first run — next run will show a before vs after comparison.)")
        return

    # Load most recent previous run
    prev_path = os.path.join(RESULTS_DIR, result_files[-1])
    with open(prev_path, "r", encoding="utf-8") as f:
        prev = json.load(f)

    print("\n" + "=" * 72)
    print("📈  BEFORE vs AFTER COMPARISON")
    print(f"    Before : {prev['timestamp']}")
    print(f"    After  : {new_summary['timestamp']}")
    print("=" * 72)
    print(f"{'Metric':<25} {'Before':>8} {'After':>8} {'Δ Change':>10}  {'Trend':>5}")
    print("─" * 62)

    for key, label in COMPARE_METRICS:
        before_val = prev.get(key, 0)
        after_val  = new_summary.get(key, 0)
        delta      = after_val - before_val

        if key == "avg_latency_ms" or key == "total_cost_usd":
            # Lower is better
            trend = "✅" if delta < 0 else ("⚠️" if delta > 0 else "➖")
        else:
            # Higher is better
            trend = "✅" if delta > 0 else ("⚠️" if delta < 0 else "➖")

        print(f"{label:<25} {before_val:>8.4f} {after_val:>8.4f} {delta:>+10.4f}  {trend:>5}")

    print("=" * 72)


# ── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary = run_evaluation()
    save_and_compare(summary)
