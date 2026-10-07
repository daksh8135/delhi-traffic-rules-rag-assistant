# src/generator.py

import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
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
    "Mvact2019.txt": "Motor Vehicles (Amendment) Act, 2019",
}

def clean_source_name(raw_filename: str) -> str:
    return SOURCE_NAMES.get(raw_filename, raw_filename)


class Generator:
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # 1. Connect to Pinecone Cloud
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pinecone_index = pc.Index("delhi-traffic-rules")

        # 2. Multilingual embedding model for incoming questions
        self.embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        # 3. Lookup table for high-frequency fine questions
        self.fine_lookup = FineLookup(os.path.join(BASE_DIR, "data", "fines_lookup.json"))

        # 4. Groq LLM
        self.llm = ChatGroq(
            model=model_name,
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

    def retrieve_chunks(self, query: str, top_k: int = 10):
        """Encodes query and retrieves top_k chunks from Pinecone."""
        query_vector = self.embed_model.encode(query).tolist()
        results = self.pinecone_index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        return [match["metadata"] for match in results["matches"]]

    def ask(self, query: str, top_k: int = 10) -> dict:
        start_time = time.time()

        # STEP 1: Quick lookup first
        quick_answer = self.fine_lookup.match(query)
        if quick_answer:
            elapsed = time.time() - start_time
            print(f"[Performance] Lookup match — total time: {elapsed:.2f}s")
            return {
                "answer": quick_answer,
                "context": "Structured fine lookup table (no retrieval used)"
            }

        # STEP 2: Query Pinecone Cloud
        retrieval_start = time.time()
        relevant_chunks = self.retrieve_chunks(query, top_k)
        retrieval_time = time.time() - retrieval_start

        context = "\n\n".join(
            [f"[Source: {clean_source_name(chunk.get('source', ''))}] {chunk['text']}" for chunk in relevant_chunks]
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
                    "If different documents cover different parts of the answer, mention both sources explicitly.\n\n"
                    "Give clear, factual, and concise answers with specific penalties, fines (₹), or legal terms.\n\n"
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
        print(f"[Performance] Pinecone: {retrieval_time:.2f}s | LLM: {llm_time:.2f}s | Total: {total_time:.2f}s")

        return {
            "answer": response.content,
            "context": context
        }


if __name__ == "__main__":
    generator = Generator()
    query = "What are the rules for under age driving?"
    result = generator.ask(query, top_k=10)
    print("\nAnswer:\n")
    print(result["answer"])