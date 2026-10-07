# src/generator.py

import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from fine_lookup import FineLookup

# Optional: Langfuse Callback for Tracing (Feature 3)
try:
    from langfuse.callback import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

load_dotenv()

SOURCE_NAMES = {
    "delhi_traffic_rules.txt": "Delhi Motor Vehicles Rules, 1993",
    "cmvr1989.txt": "Central Motor Vehicles Rules, 1989",
    "mv_act_1988.txt": "Motor Vehicles Act, 1988",
    "the_delhi_motor_vehicle_taxation_act-r.txt": "Delhi Motor Vehicle Taxation Act",
    "aA1988-59.txt": "Motor Vehicles (Amendment) Act, 2019",
    "mact.txt": "Motor Vehicles Act (Supplementary)",
    "Mvact2019.txt": "Motor Vehicles (Amendment) Act, 2019",
}

def clean_source_name(raw_filename: str) -> str:
    return SOURCE_NAMES.get(raw_filename, raw_filename)


class Generator:
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # 1. Pinecone Cloud Index
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pinecone_index = pc.Index("delhi-traffic-rules")

        # 2. Embedding Model for Queries
        self.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # 3. Lookup table for frequent fine questions
        self.fine_lookup = FineLookup(os.path.join(BASE_DIR, "data", "fines_lookup.json"))

        # 4. Groq LLM
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # 5. Langfuse Tracing Handler (Feature 3)
        self.langfuse_handler = None
        if LANGFUSE_AVAILABLE and os.getenv("LANGFUSE_PUBLIC_KEY"):
            try:
                self.langfuse_handler = CallbackHandler(
                    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                )
            except Exception as e:
                print(f"[Warning] Langfuse initialization skipped: {e}")

    def retrieve_chunks(self, query: str, top_k: int = 10):
        """Encodes query and retrieves top_k chunks from Pinecone with similarity scores."""
        query_vector = self.embed_model.encode(query).tolist()
        results = self.pinecone_index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        scored_chunks = []
        for match in results["matches"]:
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")
            scored_chunks.append({
                "id": match["id"],
                "score": round(match["score"], 4),  # Cosine similarity (0.0 to 1.0)
                "source": clean_source_name(metadata.get("source", "Unknown")),
                "snippet": (text[:160] + "...") if len(text) > 160 else text,
                "text": text
            })
        return scored_chunks

    def ask(self, query: str, top_k: int = 10) -> dict:
        start_time = time.time()

        # Step 1: Quick lookup first
        quick_answer = self.fine_lookup.match(query)
        if quick_answer:
            elapsed = time.time() - start_time
            return {
                "answer": quick_answer,
                "context": "Structured fine lookup table (no retrieval used)",
                "metrics": {
                    "total_latency_ms": round(elapsed * 1000, 2),
                    "retrieval_latency_ms": 0.0,
                    "llm_latency_ms": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "retrieved_chunks": []
                }
            }

        # Step 2: Query Pinecone (Feature 1: Latency & Similarity Scores)
        retrieval_start = time.time()
        scored_chunks = self.retrieve_chunks(query, top_k)
        retrieval_time = time.time() - retrieval_start

        context = "\n\n".join(
            [f"[Source: {c['source']}] {c['text']}" for c in scored_chunks]
        )

        messages = [
            SystemMessage(
                content=(
                    "You are a traffic law assistant trained on Delhi traffic law documents. "
                    "Answer user questions accurately using ONLY the information provided in the context.\n"
                    "Cite the exact document names when applicable and include specific fines (₹).\n\n"
                    "End with: \"Note: This information is for general awareness only. For specific legal matters, please consult a qualified legal professional or the concerned traffic authority.\""
                )
            ),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        ]

        # Step 3: LLM Inference with optional Langfuse tracing (Feature 3)
        llm_start = time.time()
        callbacks = [self.langfuse_handler] if self.langfuse_handler else []
        response = self.llm.invoke(messages, config={"callbacks": callbacks} if callbacks else None)
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time

        # Step 4: Token usage and cost calculation
        token_usage = response.response_metadata.get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens)

        # Groq LLaMA pricing: ~$0.59 / 1M prompt tokens, $0.79 / 1M completion tokens
        cost_usd = (prompt_tokens * 0.00000059) + (completion_tokens * 0.00000079)

        metrics = {
            "total_latency_ms": round(total_time * 1000, 2),
            "retrieval_latency_ms": round(retrieval_time * 1000, 2),
            "llm_latency_ms": round(llm_time * 1000, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(cost_usd, 6),
            "retrieved_chunks": scored_chunks
        }

        return {
            "answer": response.content,
            "context": context,
            "metrics": metrics
        }