# api/app.py

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from generator import Generator

app = FastAPI(
    title="Delhi Traffic Rules RAG API",
    description="Hybrid RAG backend for Delhi traffic law queries.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = Generator()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10


class QueryResponse(BaseModel):
    answer: str
    context: str


@app.get("/")
def read_root():
    return {"message": "Delhi Traffic Rules RAG API is running."}
# frontend_app.py

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Delhi Traffic Rules Assistant", page_icon="🚦")

st.title("🚦 Delhi Traffic Rules Assistant")
st.write("Ask questions about Delhi traffic rules, fines, and procedures. Supports English and Hindi.")

query = st.text_input("Your question:", placeholder="e.g. What is the fine for not wearing a helmet?")
ask_clicked = st.button("Ask", type="primary")

if query and ask_clicked:
    with st.spinner("Thinking..."):
        try:
            response = requests.post(f"{API_URL}/ask", json={"query": query, "top_k": 10})
            response.raise_for_status()
            result = response.json()
            st.markdown("### Answer")
            st.success(result["answer"])
        except requests.exceptions.ConnectionError:
            st.error("Can't reach the backend. Make sure the FastAPI server is running (uvicorn api.app:app --reload).")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

st.divider()
st.caption("Built with LangChain, FAISS, BM25 hybrid retrieval, Groq, FastAPI, and Streamlit.")

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        result = generator.ask(request.query, top_k=request.top_k)
        return {"answer": result["answer"], "context": result["context"]}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "context": ""}