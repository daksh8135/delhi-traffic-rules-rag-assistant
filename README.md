# 🚦 Delhi Traffic Rules Assistant

An AI-powered assistant built using hybrid Retrieval-Augmented Generation (RAG) that provides context-aware, cited answers about Delhi traffic rules, penalties, and driver rights. It combines dense semantic search with sparse keyword search and LLM generation for accurate, explainable, and trustworthy responses — with built-in evaluation to measure that trustworthiness, not just claim it.

**Live demo:** _[https://delhi-traffic-rules-rag-assistant-dbwuaqjm4arqjhnflatyk9.streamlit.app/]_

---

## Repo Structure
```plaintext
traffic_rules_assistant/
├── .venv/
├── data/
│   ├── raw/                          # source PDFs
│   ├── processed/                    # extracted text, chunks, FAISS + BM25 indexes
│   └── fines_lookup.json             # structured lookup table for known fines
├── eval/
│   ├── eval_dataset.json             # hand-curated test questions
│   └── eval_results.json             # latest evaluation run output
├── src/
│   ├── text_extraction.py            # PDF -> text (multi-document support)
│   ├── chunking.py                   # merges all sources into one chunk set
│   ├── embedding.py                  # builds FAISS dense index
│   ├── sparse_retriever.py           # builds BM25 sparse index
│   ├── retriever.py                  # dense retrieval
│   ├── hybrid_retriever.py           # dense + sparse fused via RRF
│   ├── fine_lookup.py                # deterministic fine lookup
│   ├── generator.py                  # grounded generation, citations, guardrails
│   ├── eval_runner.py                # LLM-as-judge evaluation harness
│   └── main.py                       # CLI chatbot
├── api/
│   └── app.py                        # Streamlit web interface
├── .gitignore
├── .env                              # GROQ_API_KEY (not committed)
├── README.md
└── requirements.txt
```

---

## Overview

This project ingests multiple official Delhi and central Motor Vehicle legal documents and makes them queryable through natural language, using a hybrid RAG pipeline built and debugged from the ground up. It uses:

- `sentence-transformers` + `FAISS` for semantic retrieval
- `rank_bm25` for keyword-based retrieval, fused with dense search via Reciprocal Rank Fusion
- `LangChain` with `Groq` LLM backend for grounded, cited response generation
- A structured lookup table for guaranteed-consistent answers on high-frequency fine queries
- A custom LLM-as-judge evaluation harness to measure — not assume — answer quality
- `Streamlit` to expose the system as a usable web interface

---

## Target Audience

- Citizens navigating Delhi traffic law
- Driving school instructors and trainees
- Traffic law educators
- AI/ML learners studying practical RAG system design and evaluation

---

## Prerequisites

- Python 3.11+
- Basic terminal/CLI knowledge
- Internet access for LLM API (Groq)
- Groq API key (free tier available at [console.groq.com](https://console.groq.com))

---

## Installation

```bash
git clone <your-repo-url>
cd traffic_rules_assistant

python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file at the project root:
```
GROQ_API_KEY=your-groq-api-key-here
```

Place source PDFs in `data/raw/` before running ingestion. This project currently ships with:
- Delhi Motor Vehicles Rules, 1993
- Central Motor Vehicles Rules, 1989
- Motor Vehicles Act, 1988
- Motor Vehicles (Amendment) Act, 2019
- Delhi Motor Vehicle Taxation Act

---

## Usage

Run each step from inside `src/`:

**1. Text extraction** (processes every PDF in `data/raw/`)
```bash
python text_extraction.py
```

**2. Chunking** (merges all extracted text into one tagged chunk set)
```bash
python chunking.py
```

**3. Embedding + FAISS index**
```bash
python embedding.py
```

**4. BM25 sparse index**
```bash
python sparse_retriever.py
```

**5. CLI chatbot**
```bash
python main.py
```

**6. Web interface**
```bash
cd ..
streamlit run api/app.py
```

**7. Run evaluation**
```bash
cd src
python eval_runner.py
```

---

## Architecture

```
        PDFs (data/raw/)
              |
              v
      Text Extraction (page-aware, multi-document)
              |
              v
      Chunking (merged, source-tagged)
              |
   +----------+----------+
   v                      v
Dense Embeddings      BM25 Keyword Index
(FAISS, MiniLM)
   +----------+----------+
              v
   Hybrid Retrieval (Reciprocal Rank Fusion)
              |
   +----------+---------------+
   |  Matches fine lookup?     |--Yes--> Instant, guaranteed answer
   +----------+----------------+
             No
              v
    LLM Generation (Groq)
    -- grounded, cited, disclaimer-appended
```

---

## Key Design Decisions

**Hybrid retrieval over pure vector search.** Dense embeddings capture semantic meaning but can under-rank chunks that share little vocabulary with the query — an issue confirmed directly during development (see below). BM25 keyword search complements this by catching exact term matches (section numbers, rupee amounts) that embeddings alone can miss.

**Structured fine lookup as a targeted mitigation.** For a small set of high-frequency violation queries, retrieval-based generation can be sensitive to exact query phrasing. A deterministic keyword-matched lookup table guarantees consistent, correct answers for these known cases, while everything else falls through to the full RAG pipeline.

**Faithfulness-first prompting.** The system is explicitly instructed to answer only from retrieved context and to decline rather than speculate when information isn't present — verified through direct, repeated testing rather than assumed.

---

## Evaluation

An automated evaluation harness (`eval_runner.py`) scores the system against a 10-question hand-curated test set — covering known fines, general RAG lookups, Hindi queries, and off-topic rejection — using an LLM judge to score each answer 1-5 on:

- **Faithfulness** — does the answer use only facts present in the retrieved context (not the judge's general knowledge)?
- **Relevance** — does the answer address the question actually asked?
- **Correctness** — does the answer align with a verified reference answer?

**Latest results:**

| Metric | Score |
|---|---|
| Faithfulness | 4.60 / 5 |
| Relevance | 5.00 / 5 |
| Correctness | 4.60 / 5 |

The judge is given the actual retrieved context alongside each answer, so faithfulness measures groundedness to retrieval specifically — an earlier version of this harness incorrectly scored faithfulness against the judge's general knowledge instead, which was identified and corrected during development.

Results are saved incrementally to `eval/eval_results.json` after every question, so a crash or rate limit mid-run doesn't discard prior results.

---

## Performance

| Path | Typical latency |
|---|---|
| Structured fine lookup | ~0.00s |
| Hybrid retrieval | 0.04s - 0.25s |
| LLM generation | 1.0s - 9.8s |
| Total (full RAG path) | 1.3s - 9.9s |

Retrieval is consistently fast; response time is dominated by LLM generation.

---

## Configuration

**Chunking:** fixed character-size chunks with overlap, tagged by source document, globally unique chunk IDs across all documents.

**Dense retrieval:** `all-MiniLM-L6-v2` embeddings, FAISS `IndexFlatIP` (cosine similarity via L2-normalized vectors).

**Sparse retrieval:** BM25 over tokenized chunk text (Unicode-aware, supports Hindi).

**Fusion:** Reciprocal Rank Fusion, `k=60`.

**LLM (Groq):** `openai/gpt-oss-20b`, temperature `0.3`.

---

## Known Limitations

- **Cross-referenced penalty clauses can be missed by retrieval.** Some legal sections state a requirement (e.g., wearing protective headgear) while the actual penalty is defined in a separate, generically-worded catch-all clause elsewhere in the same Act. Because the two share little vocabulary, both semantic and keyword retrieval can fail to connect them. Mitigated for known cases via the fine lookup table; not solved generally. A future improvement would be citation-aware retrieval that explicitly follows cross-references between sections.
- **Evaluation set is intentionally small (10 questions)** — enough to demonstrate methodology and catch real regressions during development, not yet large enough to be statistically rigorous.
- **Fine lookup table is manually curated** and should be cross-verified against official legal text before being relied on for real-world use.
- **Chunking is fixed-size**, not section-boundary-aware — a more sophisticated chunker would respect the natural structure of legal documents.

---

## Project Notes

This system was built and iterated through real debugging cycles, including:

- A query-vector normalization mismatch between index-build time and query time, which skewed retrieval rankings until identified and fixed.
- A specific test case — "what is the penalty for not wearing a helmet" — revealed that the Delhi Motor Vehicles Rules state the *requirement* in one section while the Motor Vehicles Act defines the *penalty* in a completely separate, generically-worded section. This directly motivated the hybrid retrieval architecture and the structured fine lookup table.
- An initial evaluation harness measured "faithfulness" against the judge model's general knowledge rather than the system's actual retrieved context — a subtle but important conceptual error in RAG evaluation, corrected by passing retrieved context directly into the judge prompt.

---

## License / Disclaimer

This project is for educational and portfolio purposes. Information provided by the assistant is based on publicly available legal documents and is not a substitute for professional legal advice. For specific legal matters, consult a qualified legal professional or the concerned traffic authority.

---

## Contact

**Maintainer:** _[Daksh Bains]_
- **Email:** _[dakshbains05@gmail.com]_
- **GitHub:** [@daksh8135](https://github.com/daksh8135)
**Live demo:** [https://delhi-traffic-rules-rag-assistant-dbwuaqjm4arqjhnflatyk9.streamlit.app/](https://delhi-traffic-rules-rag-assistant-dbwuaqjm4arqjhnflatyk9.streamlit.app/)