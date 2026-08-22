# src/generator.py

import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from hybrid_retriever import HybridRetriever
from fine_lookup import FineLookup

load_dotenv()

# Maps raw filenames to clean, human-readable document names for citations
SOURCE_NAMES = {
    "delhi_traffic_rules.txt": "Delhi Motor Vehicles Rules, 1993",
    "cmvr1989.txt": "Central Motor Vehicles Rules, 1989",
    "mv_act_1988.txt": "Motor Vehicles Act, 1988",
    "the_delhi_motor_vehicle_taxation_act-r.txt": "Delhi Motor Vehicle Taxation Act",
    "aA1988-59.txt": "Motor Vehicles (Amendment) Act, 2019",
    "mact.txt": "Motor Vehicles Act (Supplementary)",
    # add more entries here if you add more source PDFs later
}


def clean_source_name(raw_filename: str) -> str:
    """Converts a raw source filename into a readable document name for citations."""
    return SOURCE_NAMES.get(raw_filename, raw_filename)


class Generator:
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        self.retriever = HybridRetriever(
            dense_index_path=os.path.join(BASE_DIR, "data", "processed", "faiss_cosine_index.idx"),
            sparse_index_path=os.path.join(BASE_DIR, "data", "processed", "bm25_index.pkl"),
            chunk_json_path=os.path.join(BASE_DIR, "data", "processed", "all_chunks.json")
        )

        # Simple keyword-based lookup for known, high-frequency fine questions.
        # Checked BEFORE running full retrieval + LLM generation, so common
        # questions get a guaranteed-correct, instant answer regardless of
        # exact phrasing (which retrieval alone can be sensitive to).
        self.fine_lookup = FineLookup(os.path.join(BASE_DIR, "data", "fines_lookup.json"))

        self.llm = ChatGroq(
            model=model_name,
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def ask(self, query: str, top_k: int = 10) -> dict:
        """
        Returns a dict: {"answer": str, "context": str}
        `context` is the actual retrieved text used to produce the answer —
        needed so evaluation can check FAITHFULNESS against what the system
        actually saw, not against general knowledge.
        """
        start_time = time.time()

        # STEP 1: Check the simple lookup table first.
        quick_answer = self.fine_lookup.match(query)
        if quick_answer:
            elapsed = time.time() - start_time
            print(f"[Performance] Lookup match — total time: {elapsed:.2f}s")
            return {
                "answer": quick_answer,
                "context": "Structured fine lookup table (no retrieval used)"
            }

        # STEP 2: Fall back to full RAG pipeline for everything else.
        retrieval_start = time.time()
        relevant_chunks = self.retriever.query(query, top_k)
        retrieval_time = time.time() - retrieval_start

        context = "\n\n".join(
            [f"[Source: {clean_source_name(chunk['source'])}] {chunk['text']}" for chunk in relevant_chunks]
        )

        messages = [
            SystemMessage(
                content=(
                    "You are a traffic law assistant trained on multiple Delhi traffic law documents "
                    "(e.g. Delhi Motor Vehicles Rules, Motor Vehicles Act). "
                    "Your task is to answer user questions by using only the information provided in the context. "
                    "Do not generate answers based on external knowledge or assumptions.\n\n"

                    "If the context does not contain enough information to answer the question accurately, "
                    "politely inform the user that the answer is not available and suggest they rephrase or ask something else. "
                    "Do not mention 'context provided' or similar phrases in the answer. Do not speculate.\n\n"

                    "Multiple source documents may appear in the context, each labeled [Source: document name]. "
                    "ALWAYS cite the exact source document name shown in the context — never invent a name and "
                    "never show a raw filename. "
                    "If different documents cover different parts of the answer (e.g. one states a requirement, "
                    "another states the penalty), mention both sources explicitly.\n\n"

                    "Give clear, factual, and concise answers. If applicable, include:\n"
                    "- Specific penalties, fines (₹), or legal terms\n"
                    "- Safety instructions or procedures\n\n"

                    "You are a helpful, neutral, and official-sounding assistant for citizens, law enforcers, and learners.\n\n"

                    "Use professional, polite, and understandable language. Avoid unnecessary repetition or disclaimers.\n\n"

                    "Help users understand their rights, duties, and consequences as per Delhi road traffic laws, using only the source material.\n\n"

                    "FORMATTING RULES (important):\n"
                    "- Do NOT use markdown tables (no | or --- characters).\n"
                    "- Do NOT use bold (**text**) or markdown headers (#).\n"
                    "- Use plain text only, with simple line breaks and dashes ( - ) for lists.\n"
                    "- This output is displayed in a plain terminal that cannot render markdown, so formatted "
                    "markdown will look broken and messy — always avoid it.\n\n"

                    "DISCLAIMER: End every answer with this exact line on its own:\n"
                    "\"Note: This information is for general awareness only. For specific legal matters, please consult a qualified legal professional or the concerned traffic authority.\""
                )
            ),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {query}"
            )
        ]

        llm_start = time.time()
        response = self.llm.invoke(messages)
        llm_time = time.time() - llm_start

        total_time = time.time() - start_time
        print(f"[Performance] Retrieval: {retrieval_time:.2f}s | LLM: {llm_time:.2f}s | Total: {total_time:.2f}s")

        return {
            "answer": response.content,
            "context": context
        }


if __name__ == "__main__":
    generator = Generator()
    query = "What are the rules for overtaking?"
    result = generator.ask(query, top_k=20)
    print("\nAnswer:\n")
    print(result["answer"])